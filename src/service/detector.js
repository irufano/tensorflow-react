import * as tfconv from "@tensorflow/tfjs-converter";
import * as tf from "@tensorflow/tfjs-core";

/**
 * CustomFaceDetector Class
 *
 * @constructor detectorModel
 * @constructor maxFaces
 * @constructor minScoreThresh default 0.5
 *
 */
export class CustomFaceDetector {
  constructor(detectorModel, maxFaces, minScoreThresh = 0.5) {
    this.detectorModel = detectorModel;
    this.maxFaces = maxFaces;
    this.imageToTensorConfig = SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG;
    this.tensorsToDetectionConfig = SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG;
    this.tensorsToDetectionConfig.minScoreThresh = minScoreThresh;
    this.anchors = createSsdAnchors(SHORT_RANGE_DETECTOR_ANCHOR_CONFIG);

    const anchorW = tf.tensor1d(this.anchors.map((a) => a.width));
    const anchorH = tf.tensor1d(this.anchors.map((a) => a.height));
    const anchorX = tf.tensor1d(this.anchors.map((a) => a.xCenter));
    const anchorY = tf.tensor1d(this.anchors.map((a) => a.yCenter));
    this.anchorTensor = { x: anchorX, y: anchorY, w: anchorW, h: anchorH };
  }

  /**
   * dispose
   * 
   */
  dispose() {
    this.detectorModel.dispose();
    tf.dispose([
      this.anchorTensor.x,
      this.anchorTensor.y,
      this.anchorTensor.w,
      this.anchorTensor.h,
    ]);
  }

  /**
   * reset
   * 
   */
  reset() {}

  /**
   * Detects faces.
   *
   * Subgraph: FaceDetectionShort/FullRangeCpu.
   *
   * ref:
   * https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt
   * https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_full_range_cpu.pbtxt
   */
  async detectFaces(input, flipHorizontal = false) {
    if (input === null) {
      this.reset();
      return [];
    }

    const image3d = tf.tidy(() => {
      let imageTensor = tf.cast(toImageTensor(input), "float32");
      if (flipHorizontal) {
        const batchAxis = 0;
        imageTensor = tf.squeeze(
          tf.image.flipLeftRight(
            // eslint-disable-next-line
            tf.expandDims(imageTensor, batchAxis)
          ),
          [batchAxis]
        );
      }
      return imageTensor;
    });

    // FaceDetectionShort/FullRangeModelCpu: ImageToTensorCalculator
    // Transforms the input image into a 128x128 tensor while keeping the aspect
    // ratio (what is expected by the corresponding face detection model),
    // resulting in potential letterboxing in the transformed image.
    const { imageTensor, transformationMatrix } = convertImageToTensor(
      image3d,
      this.imageToTensorConfig
    );

    const detectionResult = this.detectorModel.execute(
      imageTensor,
      "Identity:0"
    );
    // FaceDetectionShort/FullRangeModelCpu: InferenceCalculator
    // The model returns a tensor with the following shape:
    // [1 (batch), 896 (anchor points), 17 (data for each anchor)]
    const { boxes, logits } = detectorResult(detectionResult);
    // FaceDetectionShort/FullRangeModelCpu: TensorsToDetectionsCalculator
    const unfilteredDetections = await tensorsToDetections(
      [logits, boxes],
      this.anchorTensor,
      this.tensorsToDetectionConfig
    );

    if (unfilteredDetections.length === 0) {
      tf.dispose([image3d, imageTensor, detectionResult, logits, boxes]);
      return unfilteredDetections;
    }

    // FaceDetectionShort/FullRangeModelCpu: NonMaxSuppressionCalculator
    const filteredDetections = await nonMaxSuppression(
      unfilteredDetections,
      this.maxFaces,
      DETECTOR_NON_MAX_SUPPRESSION_CONFIG.minSuppressionThreshold,
      DETECTOR_NON_MAX_SUPPRESSION_CONFIG.overlapType
    );

    const detections =
      // FaceDetectionShortRangeModelCpu:
      // DetectionProjectionCalculator
      detectionProjection(filteredDetections, transformationMatrix);

    tf.dispose([image3d, imageTensor, detectionResult, logits, boxes]);

    // console.log(detections);
    return detections;
  }

  /**
   * estimateFaces function
   *
   * @param input input image/video
   * @param estimationConfig estimationConfig.flipHorizontal default false
   *
   */
  async estimateFaces(input, estimationConfig = undefined) {
    const imageSize = getImageSize(input);
    const flipHorizontal = estimationConfig
      ? estimationConfig.flipHorizontal
      : false;

    return this.detectFaces(input, flipHorizontal).then((detections) =>
      detections.map((detection) => {
        const score = detection.score[0] ?? 0;
        const keypoints = detection.locationData.relativeKeypoints.map(
          (keypoint, i) => ({
            ...keypoint,
            x: keypoint.x * imageSize.width,
            y: keypoint.y * imageSize.height,
            name: MEDIAPIPE_FACE_DETECTOR_KEYPOINTS[i],
          })
        );
        const box = detection.locationData.relativeBoundingBox;

        for (const key of ["width", "xMax", "xMin"]) {
          box[key] *= imageSize.width;
        }
        for (const key of ["height", "yMax", "yMin"]) {
          box[key] *= imageSize.height;
        }
        return { score, keypoints, box };
      })
    );
  }
}

/**
 * createFaceDetector function
 *
 * @param maxFaces default 1
 * @param minScoreThresh default 0.5
 * @param onProgress default (progress) => {}
 *
 */
export async function createFaceDetector({
  maxFaces = 1,
  minScoreThresh = 0.5,
  onProgress = (_) => {},
} = {}) {
  const model = await loadModel(onProgress);
  return new CustomFaceDetector(model, maxFaces, minScoreThresh);
}

/**
 * loadModel function
 *
 * check first on caches if model exist on caches load from cachedModel
 * if not exist on cached load again from path
 *
 * @param onProgress default (progress) => {}
 *
 */
export async function loadModel({ onProgress = (progress) => {} } = {}) {
  const modelCacheKey = "fd-short";
  const modelCheckKey = "tensorflowjs_models/fd-short/info";
  // const modelPath = `${process.env.PUBLIC_URL}/models/face-short/`;
  const modelPath = `https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1`;
  let isCached = false;
  let model;

  // Check if the model is already in localStorage
  isCached = localStorage.getItem(modelCheckKey) !== null;

  if (isCached) {
    // If cached
    model = await tfconv.loadGraphModel(`localstorage://${modelCacheKey}`);
  } else {
    // If not cached, load a pre-trained face detection model
    const newModel = await tfconv.loadGraphModel(modelPath, {
      fromTFHub: true,
      onProgress: onProgress,
    });

    // save to browser cached local storage
    await newModel.save(`localstorage://${modelCacheKey}`);
    model = newModel;
  }

  return model;
}

const SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG = {
  outputTensorSize: { width: 128, height: 128 },
  keepAspectRatio: true,
  outputTensorFloatRange: [-1, 1],
  borderMode: "zero",
};

const SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG = {
  applyExponentialOnBoxSize: false,
  flipVertically: false,
  ignoreClasses: [],
  numClasses: 1,
  numBoxes: 896,
  numCoords: 16,
  boxCoordOffset: 0,
  keypointCoordOffset: 4,
  numKeypoints: 6,
  numValuesPerKeypoint: 2,
  sigmoidScore: true,
  scoreClippingThresh: 100.0,
  reverseOutputOrder: true,
  xScale: 128.0,
  yScale: 128.0,
  hScale: 128.0,
  wScale: 128.0,
  minScoreThresh: 0.5,
};

const SHORT_RANGE_DETECTOR_ANCHOR_CONFIG = {
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 1.0,
  featureMapHeight: [],
  featureMapWidth: [],
  numLayers: 4,
  minScale: 0.1484375,
  maxScale: 0.75,
  inputSizeHeight: 128,
  inputSizeWidth: 128,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [8, 16, 16, 16],
  aspectRatios: [1.0],
  fixedAnchorSize: true,
};

const DETECTOR_NON_MAX_SUPPRESSION_CONFIG = {
  overlapType: "intersection-over-union",
  minSuppressionThreshold: 0.3,
};

const MEDIAPIPE_FACE_DETECTOR_KEYPOINTS = [
  "rightEye",
  "leftEye",
  "noseTip",
  "mouthCenter",
  "rightEarTragion",
  "leftEarTragion",
];

/**
 * Convert an image to an image tensor representation.
 *
 * The image tensor has a shape [1, height, width, colorChannel].
 *
 * @param input An image, video frame, or image tensor.
 */
function toImageTensor(input) {
  return input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
}

// ref:
// https://github.com/google/mediapipe/blob/350fbb2100ad531bc110b93aaea23d96af5a5064/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
function createSsdAnchors(config) {
  // Set defaults.
  if (config.reduceBoxesInLowestLayer == null) {
    config.reduceBoxesInLowestLayer = false;
  }
  if (config.interpolatedScaleAspectRatio == null) {
    config.interpolatedScaleAspectRatio = 1.0;
  }
  if (config.fixedAnchorSize == null) {
    config.fixedAnchorSize = false;
  }

  const anchors = [];
  let layerId = 0;
  while (layerId < config.numLayers) {
    const anchorHeight = [];
    const anchorWidth = [];
    const aspectRatios = [];
    const scales = [];

    // For same strides, we merge the anchors in the same order.
    let lastSameStrideLayer = layerId;
    while (
      lastSameStrideLayer < config.strides.length &&
      config.strides[lastSameStrideLayer] === config.strides[layerId]
    ) {
      const scale = calculateScale(
        config.minScale,
        config.maxScale,
        lastSameStrideLayer,
        config.strides.length
      );
      if (lastSameStrideLayer === 0 && config.reduceBoxesInLowestLayer) {
        // For first layer, it can be specified to use predefined anchors.
        aspectRatios.push(1);
        aspectRatios.push(2);
        aspectRatios.push(0.5);
        scales.push(0.1);
        scales.push(scale);
        scales.push(scale);
      } else {
        for (
          let aspectRatioId = 0;
          aspectRatioId < config.aspectRatios.length;
          ++aspectRatioId
        ) {
          aspectRatios.push(config.aspectRatios[aspectRatioId]);
          scales.push(scale);
        }
        if (config.interpolatedScaleAspectRatio > 0.0) {
          const scaleNext =
            lastSameStrideLayer === config.strides.length - 1
              ? 1.0
              : calculateScale(
                  config.minScale,
                  config.maxScale,
                  lastSameStrideLayer + 1,
                  config.strides.length
                );
          scales.push(Math.sqrt(scale * scaleNext));
          aspectRatios.push(config.interpolatedScaleAspectRatio);
        }
      }
      lastSameStrideLayer++;
    }

    for (let i = 0; i < aspectRatios.length; ++i) {
      const ratioSqrts = Math.sqrt(aspectRatios[i]);
      anchorHeight.push(scales[i] / ratioSqrts);
      anchorWidth.push(scales[i] * ratioSqrts);
    }

    let featureMapHeight = 0;
    let featureMapWidth = 0;
    if (config.featureMapHeight.length > 0) {
      featureMapHeight = config.featureMapHeight[layerId];
      featureMapWidth = config.featureMapWidth[layerId];
    } else {
      const stride = config.strides[layerId];
      featureMapHeight = Math.ceil(config.inputSizeHeight / stride);
      featureMapWidth = Math.ceil(config.inputSizeWidth / stride);
    }

    for (let y = 0; y < featureMapHeight; ++y) {
      for (let x = 0; x < featureMapWidth; ++x) {
        for (let anchorId = 0; anchorId < anchorHeight.length; ++anchorId) {
          const xCenter = (x + config.anchorOffsetX) / featureMapWidth;
          const yCenter = (y + config.anchorOffsetY) / featureMapHeight;

          const newAnchor = { xCenter, yCenter, width: 0, height: 0 };

          if (config.fixedAnchorSize) {
            newAnchor.width = 1.0;
            newAnchor.height = 1.0;
          } else {
            newAnchor.width = anchorWidth[anchorId];
            newAnchor.height = anchorHeight[anchorId];
          }
          anchors.push(newAnchor);
        }
      }
    }
    layerId = lastSameStrideLayer;
  }

  return anchors;
}

function calculateScale(minScale, maxScale, strideIndex, numStrides) {
  if (numStrides === 1) {
    return (minScale + maxScale) * 0.5;
  } else {
    return minScale + ((maxScale - minScale) * strideIndex) / (numStrides - 1);
  }
}

/**
 * Convert an image or part of it to an image tensor.
 *
 * @param image An image, video frame or image tensor.
 * @param config
 *      inputResolution: The target height and width.
 *      keepAspectRatio?: Whether target tensor should keep aspect ratio.
 * @param normRect A normalized rectangle, representing the subarea to crop from
 *      the image. If normRect is provided, the returned image tensor represents
 *      the subarea.
 * @returns A map with the following properties:
 *     - imageTensor
 *     - padding: Padding ratio of left, top, right, bottom, based on the output
 * dimensions.
 *     - transformationMatrix: Projective transform matrix used to transform
 * input image to transformed image.
 */
function convertImageToTensor(image, config, normRect = undefined) {
  const {
    outputTensorSize,
    keepAspectRatio,
    borderMode,
    outputTensorFloatRange,
  } = config;

  // Ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tensor/image_to_tensor_calculator.cc
  const imageSize = getImageSize(image);
  const roi = getRoi(imageSize, normRect);
  const padding = padRoi(roi, outputTensorSize, keepAspectRatio);
  const transformationMatrix = getRotatedSubRectToRectTransformMatrix(
    roi,
    imageSize.width,
    imageSize.height,
    false
  );

  const imageTensor = tf.tidy(() => {
    const $image = toImageTensor(image);

    const transformMatrix = tf.tensor2d(
      getProjectiveTransformMatrix(
        transformationMatrix,
        imageSize,
        outputTensorSize
      ),
      [1, 8]
    );

    const fillMode = borderMode === "zero" ? "constant" : "nearest";

    const imageTransformed = tf.image.transform(
      // tslint:disable-next-line: no-unnecessary-type-assertion
      tf.expandDims(tf.cast($image, "float32")),
      transformMatrix,
      "bilinear",
      fillMode,
      0,
      [outputTensorSize.height, outputTensorSize.width]
    );

    const imageShifted =
      outputTensorFloatRange != null
        ? shiftImageValue(imageTransformed, outputTensorFloatRange)
        : imageTransformed;

    return imageShifted;
  });

  return { imageTensor, padding, transformationMatrix };
}

function getImageSize(input) {
  if (input instanceof tf.Tensor) {
    return { height: input.shape[0], width: input.shape[1] };
  } else {
    return { height: input.height, width: input.width };
  }
}

/**
 * Get the rectangle information of an image, including xCenter, yCenter, width,
 * height and rotation.
 *
 * @param imageSize imageSize is used to calculate the rectangle.
 * @param normRect Optional. If normRect is not null, it will be used to get
 *     a subarea rectangle information in the image. `imageSize` is used to
 *     calculate the actual non-normalized coordinates.
 */
function getRoi(imageSize, normRect) {
  if (normRect) {
    return {
      xCenter: normRect.xCenter * imageSize.width,
      yCenter: normRect.yCenter * imageSize.height,
      width: normRect.width * imageSize.width,
      height: normRect.height * imageSize.height,
      rotation: normRect.rotation,
    };
  } else {
    return {
      xCenter: 0.5 * imageSize.width,
      yCenter: 0.5 * imageSize.height,
      width: imageSize.width,
      height: imageSize.height,
      rotation: 0,
    };
  }
}

/**
 * Padding ratio of left, top, right, bottom, based on the output dimensions.
 *
 * The padding values are non-zero only when the "keep_aspect_ratio" is true.
 *
 * For instance, when the input image is 10x10 (width x height) and the
 * output dimensions is 20x40 and "keep_aspect_ratio" is true, we should scale
 * the input image to 20x20 and places it in the middle of the output image with
 * an equal padding of 10 pixels at the top and the bottom. The result is
 * therefore {left: 0, top: 0.25, right: 0, bottom: 0.25} (10/40 = 0.25f).
 * @param roi The original rectangle to pad.
 * @param targetSize The target width and height of the result rectangle.
 * @param keepAspectRatio Whether keep aspect ratio. Default to false.
 */
function padRoi(roi, targetSize, keepAspectRatio = false) {
  if (!keepAspectRatio) {
    return { top: 0, left: 0, right: 0, bottom: 0 };
  }

  const targetH = targetSize.height;
  const targetW = targetSize.width;

  validateSize(targetSize, "targetSize");
  validateSize(roi, "roi");

  const tensorAspectRatio = targetH / targetW;
  const roiAspectRatio = roi.height / roi.width;
  let newWidth;
  let newHeight;
  let horizontalPadding = 0;
  let verticalPadding = 0;
  if (tensorAspectRatio > roiAspectRatio) {
    // pad height;
    newWidth = roi.width;
    newHeight = roi.width * tensorAspectRatio;
    verticalPadding = (1 - roiAspectRatio / tensorAspectRatio) / 2;
  } else {
    // pad width.
    newWidth = roi.height / tensorAspectRatio;
    newHeight = roi.height;
    horizontalPadding = (1 - tensorAspectRatio / roiAspectRatio) / 2;
  }

  roi.width = newWidth;
  roi.height = newHeight;

  return {
    top: verticalPadding,
    left: horizontalPadding,
    right: horizontalPadding,
    bottom: verticalPadding,
  };
}

function validateSize(size, name) {
  tf.util.assert(size.width !== 0, () => `${name} width cannot be 0.`);
  tf.util.assert(size.height !== 0, () => `${name} height cannot be 0.`);
}

/**
 * Generates a 4x4 projective transform matrix M, so that for any point in the
 * subRect image p(x, y), we can use the matrix to calculate the projected point
 * in the original image p' (x', y'): p' = p * M;
 *
 * @param subRect Rotated sub rect in absolute coordinates.
 * @param rectWidth
 * @param rectHeight
 * @param flipHorizontaly Whether to flip the image horizontally.
 */
// Ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tensor/image_to_tensor_utils.h
function getRotatedSubRectToRectTransformMatrix(
  subRect,
  rectWidth,
  rectHeight,
  flipHorizontally
) {
  // The resulting matrix is multiplication of below commented out matrices:
  //   postScaleMatrix
  //     * translateMatrix
  //     * rotateMatrix
  //     * flipMatrix
  //     * scaleMatrix
  //     * initialTranslateMatrix

  // For any point in the transformed image p, we can use the above matrix to
  // calculate the projected point in the original image p'. So that:
  // p' = p * M;
  // Note: The transform matrix below assumes image coordinates is normalized
  // to [0, 1] range.

  // Matrix to convert X,Y to [-0.5, 0.5] range "initialTranslateMatrix"
  // [ 1.0,  0.0, 0.0, -0.5]
  // [ 0.0,  1.0, 0.0, -0.5]
  // [ 0.0,  0.0, 1.0,  0.0]
  // [ 0.0,  0.0, 0.0,  1.0]

  const a = subRect.width;
  const b = subRect.height;
  // Matrix to scale X,Y,Z to sub rect "scaleMatrix"
  // Z has the same scale as X.
  // [   a, 0.0, 0.0, 0.0]
  // [0.0,    b, 0.0, 0.0]
  // [0.0, 0.0,    a, 0.0]
  // [0.0, 0.0, 0.0, 1.0]

  const flip = flipHorizontally ? -1 : 1;
  // Matrix for optional horizontal flip around middle of output image.
  // [ fl  , 0.0, 0.0, 0.0]
  // [ 0.0, 1.0, 0.0, 0.0]
  // [ 0.0, 0.0, 1.0, 0.0]
  // [ 0.0, 0.0, 0.0, 1.0]

  const c = Math.cos(subRect.rotation);
  const d = Math.sin(subRect.rotation);
  // Matrix to do rotation around Z axis "rotateMatrix"
  // [    c,   -d, 0.0, 0.0]
  // [    d,    c, 0.0, 0.0]
  // [ 0.0, 0.0, 1.0, 0.0]
  // [ 0.0, 0.0, 0.0, 1.0]

  const e = subRect.xCenter;
  const f = subRect.yCenter;
  // Matrix to do X,Y translation of sub rect within parent rect
  // "translateMatrix"
  // [1.0, 0.0, 0.0, e   ]
  // [0.0, 1.0, 0.0, f   ]
  // [0.0, 0.0, 1.0, 0.0]
  // [0.0, 0.0, 0.0, 1.0]

  const g = 1.0 / rectWidth;
  const h = 1.0 / rectHeight;
  // Matrix to scale X,Y,Z to [0.0, 1.0] range "postScaleMatrix"
  // [g,    0.0, 0.0, 0.0]
  // [0.0, h,    0.0, 0.0]
  // [0.0, 0.0,    g, 0.0]
  // [0.0, 0.0, 0.0, 1.0]

  const matrix = new Array(16);
  // row 1
  matrix[0] = a * c * flip * g;
  matrix[1] = -b * d * g;
  matrix[2] = 0.0;
  matrix[3] = (-0.5 * a * c * flip + 0.5 * b * d + e) * g;

  // row 2
  matrix[4] = a * d * flip * h;
  matrix[5] = b * c * h;
  matrix[6] = 0.0;
  matrix[7] = (-0.5 * b * c - 0.5 * a * d * flip + f) * h;

  // row 3
  matrix[8] = 0.0;
  matrix[9] = 0.0;
  matrix[10] = a * g;
  matrix[11] = 0.0;

  // row 4
  matrix[12] = 0.0;
  matrix[13] = 0.0;
  matrix[14] = 0.0;
  matrix[15] = 1.0;

  return arrayToMatrix4x4(matrix);
}

function arrayToMatrix4x4(array) {
  if (array.length !== 16) {
    throw new Error(`Array length must be 16 but got ${array.length}`);
  }
  return [
    [array[0], array[1], array[2], array[3]],
    [array[4], array[5], array[6], array[7]],
    [array[8], array[9], array[10], array[11]],
    [array[12], array[13], array[14], array[15]],
  ];
}

/**
 * Generate the projective transformation matrix to be used for `tf.transform`.
 *
 * See more documentation in `tf.transform`.
 *
 * @param matrix The transformation matrix mapping subRect to rect, can be
 *     computed using `getRotatedSubRectToRectTransformMatrix` calculator.
 * @param imageSize The original image height and width.
 * @param inputResolution The target height and width.
 */
function getProjectiveTransformMatrix(matrix, imageSize, inputResolution) {
  validateSize(inputResolution, "inputResolution");

  // To use M with regular x, y coordinates, we need to normalize them first.
  // Because x' = a0 * x + a1 * y + a2, y' = b0 * x + b1 * y + b2,
  // we need to use factor (1/inputResolution.width) to normalize x for a0 and
  // b0, similarly we need to use factor (1/inputResolution.height) to normalize
  // y for a1 and b1.
  // Also at the end, we need to de-normalize x' and y' to regular coordinates.
  // So we need to use factor imageSize.width for a0, a1 and a2, similarly
  // we need to use factor imageSize.height for b0, b1 and b2.
  const a0 = (1 / inputResolution.width) * matrix[0][0] * imageSize.width;
  const a1 = (1 / inputResolution.height) * matrix[0][1] * imageSize.width;
  const a2 = matrix[0][3] * imageSize.width;
  const b0 = (1 / inputResolution.width) * matrix[1][0] * imageSize.height;
  const b1 = (1 / inputResolution.height) * matrix[1][1] * imageSize.height;
  const b2 = matrix[1][3] * imageSize.height;

  return [a0, a1, a2, b0, b1, b2, 0, 0];
}

function shiftImageValue(image, outputFloatRange) {
  // Calculate the scale and offset to shift from [0, 255] to [-1, 1].
  const valueRange = transformValueRange(
    0,
    255,
    outputFloatRange[0] /* min */,
    outputFloatRange[1] /* max */
  );

  // Shift value range.
  return tf.tidy(() =>
    tf.add(tf.mul(image, valueRange.scale), valueRange.offset)
  );
}

function transformValueRange(fromMin, fromMax, toMin, toMax) {
  const fromRange = fromMax - fromMin;
  const toRange = toMax - toMin;

  if (fromRange === 0) {
    throw new Error(
      `Original min and max are both ${fromMin}, range cannot be 0.`
    );
  }

  const scale = toRange / fromRange;
  const offset = toMin - fromMin * scale;
  return { scale, offset };
}

function detectorResult(detectionResult) {
  return tf.tidy(() => {
    const [logits, rawBoxes] = splitDetectionResult(detectionResult);
    // Shape [896, 12]
    const rawBoxes2d = tf.squeeze(rawBoxes);
    // Shape [896]
    const logits1d = tf.squeeze(logits);

    return { boxes: rawBoxes2d, logits: logits1d };
  });
}

function splitDetectionResult(detectionResult) {
  return tf.tidy(() => {
    // logit is stored in the first element in each anchor data.
    const logits = tf.slice(detectionResult, [0, 0, 0], [1, -1, 1]);
    // Bounding box coords are stored in the next four elements for each anchor
    // point.
    const rawBoxes = tf.slice(detectionResult, [0, 0, 1], [1, -1, -1]);

    return [logits, rawBoxes];
  });
}

/**
 * Convert result Tensors from object detection models into Detection boxes.
 *
 * @param detectionTensors List of Tensors of type Float32. The list of tensors
 *     can have 2 or 3 tensors. First tensor is the predicted raw
 *     boxes/keypoints. The size of the values must be
 *     (num_boxes * num_predicted_values). Second tensor is the score tensor.
 *     The size of the valuse must be (num_boxes * num_classes). It's optional
 *     to pass in a third tensor for anchors (e.g. for SSD models) depend on the
 *     outputs of the detection model. The size of anchor tensor must be
 *     (num_boxes * 4).
 * @param anchor A tensor for anchors. The size of anchor tensor must be
 *     (num_boxes * 4).
 * @param config
 */
async function tensorsToDetections(detectionTensors, anchor, config) {
  const rawScoreTensor = detectionTensors[0];
  const rawBoxTensor = detectionTensors[1];

  // Shape [numOfBoxes, 4] or [numOfBoxes, 12].
  const boxes = decodeBoxes(rawBoxTensor, anchor, config);

  // Filter classes by scores.
  const normalizedScore = tf.tidy(() => {
    let normalizedScore = rawScoreTensor;
    if (config.sigmoidScore) {
      if (config.scoreClippingThresh != null) {
        normalizedScore = tf.clipByValue(
          rawScoreTensor,
          -config.scoreClippingThresh,
          config.scoreClippingThresh
        );
      }
      normalizedScore = tf.sigmoid(normalizedScore);
      return normalizedScore;
    }

    return normalizedScore;
  });

  const outputDetections = await convertToDetections(
    boxes,
    normalizedScore,
    config
  );

  tf.dispose([boxes, normalizedScore]);

  return outputDetections;
}

async function convertToDetections(detectionBoxes, detectionScore, config) {
  const outputDetections = [];
  const detectionBoxesData = await detectionBoxes.data();
  const detectionScoresData = await detectionScore.data();

  for (let i = 0; i < config.numBoxes; ++i) {
    if (
      config.minScoreThresh != null &&
      detectionScoresData[i] < config.minScoreThresh
    ) {
      continue;
    }
    const boxOffset = i * config.numCoords;
    const detection = convertToDetection(
      detectionBoxesData[boxOffset + 0] /* boxYMin */,
      detectionBoxesData[boxOffset + 1] /* boxXMin */,
      detectionBoxesData[boxOffset + 2] /* boxYMax */,
      detectionBoxesData[boxOffset + 3] /* boxXMax */,
      detectionScoresData[i],
      config.flipVertically,
      i
    );
    const bbox = detection.locationData.relativeBoundingBox;

    if (bbox.width < 0 || bbox.height < 0) {
      // Decoded detection boxes could have negative values for width/height
      // due to model prediction. Filter out those boxes since some
      // downstream calculators may assume non-negative values.
      continue;
    }
    // Add keypoints.
    if (config.numKeypoints > 0) {
      const locationData = detection.locationData;
      locationData.relativeKeypoints = [];
      const totalIdx = config.numKeypoints * config.numValuesPerKeypoint;
      for (let kpId = 0; kpId < totalIdx; kpId += config.numValuesPerKeypoint) {
        const keypointIndex = boxOffset + config.keypointCoordOffset + kpId;
        const keypoint = {
          x: detectionBoxesData[keypointIndex + 0],
          y: config.flipVertically
            ? 1 - detectionBoxesData[keypointIndex + 1]
            : detectionBoxesData[keypointIndex + 1],
        };
        locationData.relativeKeypoints.push(keypoint);
      }
    }
    outputDetections.push(detection);
  }

  return outputDetections;
}

function convertToDetection(
  boxYMin,
  boxXMin,
  boxYMax,
  boxXMax,
  score,
  flipVertically,
  i
) {
  return {
    score: [score],
    ind: i,
    locationData: {
      relativeBoundingBox: {
        xMin: boxXMin,
        yMin: flipVertically ? 1 - boxYMax : boxYMin,
        xMax: boxXMax,
        yMax: flipVertically ? 1 - boxYMin : boxYMax,
        width: boxXMax - boxXMin,
        height: boxYMax - boxYMin,
      },
    },
  };
}

//[xCenter, yCenter, w, h, kp1, kp2, kp3, kp4]
//[yMin, xMin, yMax, xMax, kpX, kpY, kpX, kpY]
function decodeBoxes(rawBoxes, anchor, config) {
  return tf.tidy(() => {
    let yCenter;
    let xCenter;
    let h;
    let w;

    if (config.reverseOutputOrder) {
      // Shape [numOfBoxes, 1].
      xCenter = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 0], [-1, 1])
      );
      yCenter = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 1], [-1, 1])
      );
      w = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 2], [-1, 1])
      );
      h = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 3], [-1, 1])
      );
    } else {
      yCenter = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 0], [-1, 1])
      );
      xCenter = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 1], [-1, 1])
      );
      h = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 2], [-1, 1])
      );
      w = tf.squeeze(
        tf.slice(rawBoxes, [0, config.boxCoordOffset + 3], [-1, 1])
      );
    }

    xCenter = tf.add(
      tf.mul(tf.div(xCenter, config.xScale), anchor.w),
      anchor.x
    );
    yCenter = tf.add(
      tf.mul(tf.div(yCenter, config.yScale), anchor.h),
      anchor.y
    );

    if (config.applyExponentialOnBoxSize) {
      h = tf.mul(tf.exp(tf.div(h, config.hScale)), anchor.h);
      w = tf.mul(tf.exp(tf.div(w, config.wScale)), anchor.w);
    } else {
      h = tf.mul(tf.div(h, config.hScale), anchor.h);
      w = tf.mul(tf.div(w, config.wScale), anchor.h);
    }

    const yMin = tf.sub(yCenter, tf.div(h, 2));
    const xMin = tf.sub(xCenter, tf.div(w, 2));
    const yMax = tf.add(yCenter, tf.div(h, 2));
    const xMax = tf.add(xCenter, tf.div(w, 2));

    // Shape [numOfBoxes, 4].
    let boxes = tf.concat(
      [
        tf.reshape(yMin, [config.numBoxes, 1]),
        tf.reshape(xMin, [config.numBoxes, 1]),
        tf.reshape(yMax, [config.numBoxes, 1]),
        tf.reshape(xMax, [config.numBoxes, 1]),
      ],
      1
    );

    if (config.numKeypoints) {
      for (let k = 0; k < config.numKeypoints; ++k) {
        const keypointOffset =
          config.keypointCoordOffset + k * config.numValuesPerKeypoint;
        let keypointX;
        let keypointY;
        if (config.reverseOutputOrder) {
          keypointX = tf.squeeze(
            tf.slice(rawBoxes, [0, keypointOffset], [-1, 1])
          );
          keypointY = tf.squeeze(
            tf.slice(rawBoxes, [0, keypointOffset + 1], [-1, 1])
          );
        } else {
          keypointY = tf.squeeze(
            tf.slice(rawBoxes, [0, keypointOffset], [-1, 1])
          );
          keypointX = tf.squeeze(
            tf.slice(rawBoxes, [0, keypointOffset + 1], [-1, 1])
          );
        }
        const keypointXNormalized = tf.add(
          tf.mul(tf.div(keypointX, config.xScale), anchor.w),
          anchor.x
        );
        const keypointYNormalized = tf.add(
          tf.mul(tf.div(keypointY, config.yScale), anchor.h),
          anchor.y
        );
        boxes = tf.concat(
          [
            boxes,
            tf.reshape(keypointXNormalized, [config.numBoxes, 1]),
            tf.reshape(keypointYNormalized, [config.numBoxes, 1]),
          ],
          1
        );
      }
    }

    // Shape [numOfBoxes, 4] || [numOfBoxes, 12].
    return boxes;
  });
}

async function nonMaxSuppression(
  detections,
  maxDetections,
  iouThreshold,
  // Currently only IOU overap is supported.
  overlapType = "intersection-over-union"
) {
  // Sort to match NonMaxSuppresion calculator's decreasing detection score
  // traversal.
  // NonMaxSuppresionCalculator: RetainMaxScoringLabelOnly
  detections.sort(
    (detectionA, detectionB) =>
      Math.max(...detectionB.score) - Math.max(...detectionA.score)
  );

  const detectionsTensor = tf.tensor2d(
    detections.map((d) => [
      d.locationData.relativeBoundingBox.yMin,
      d.locationData.relativeBoundingBox.xMin,
      d.locationData.relativeBoundingBox.yMax,
      d.locationData.relativeBoundingBox.xMax,
    ])
  );
  const scoresTensor = tf.tensor1d(detections.map((d) => d.score[0]));

  const selectedIdsTensor = await tf.image.nonMaxSuppressionAsync(
    detectionsTensor,
    scoresTensor,
    maxDetections,
    iouThreshold
  );
  const selectedIds = await selectedIdsTensor.array();

  const selectedDetections = detections.filter(
    (_, i) => selectedIds.indexOf(i) > -1
  );

  tf.dispose([detectionsTensor, scoresTensor, selectedIdsTensor]);

  return selectedDetections;
}

/**
 * Projects detections to a different coordinate system using a provided
 * projection matrix.
 *
 * @param detections A list of detections to project using the provided
 *     projection matrix.
 * @param projectionMatrix Maps data from one coordinate system to     another.
 * @returns detections: A list of projected detections
 */
// ref:
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/detection_projection_calculator.cc
function detectionProjection(detections = [], projectionMatrix) {
  const flatProjectionMatrix = matrix4x4ToArray(projectionMatrix);

  detections.forEach((detection) => {
    const locationData = detection.locationData;

    // Project keypoints.
    locationData.relativeKeypoints.forEach((keypoint) => {
      const [x, y] = project(flatProjectionMatrix, [keypoint.x, keypoint.y]);
      keypoint.x = x;
      keypoint.y = y;
    });

    // Project bounding box.
    const box = locationData.relativeBoundingBox;

    let xMin = Number.MAX_VALUE,
      yMin = Number.MAX_VALUE,
      xMax = Number.MIN_VALUE,
      yMax = Number.MIN_VALUE;

    [
      [box.xMin, box.yMin],
      [box.xMin + box.width, box.yMin],
      [box.xMin + box.width, box.yMin + box.height],
      [box.xMin, box.yMin + box.height],
    ].forEach((coordinate) => {
      // a) Define and project box points.
      const [x, y] = project(flatProjectionMatrix, coordinate);
      // b) Find new left top and right bottom points for a box which
      // encompases
      //    non-projected (rotated) box.
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    });
    locationData.relativeBoundingBox = {
      xMin,
      xMax,
      yMin,
      yMax,
      width: xMax - xMin,
      height: yMax - yMin,
    };
  });

  return detections;
}

function project(projectionMatrix, [x, y]) {
  return [
    x * projectionMatrix[0] + y * projectionMatrix[1] + projectionMatrix[3],
    x * projectionMatrix[4] + y * projectionMatrix[5] + projectionMatrix[7],
  ];
}

function matrix4x4ToArray(matrix) {
  return [].concat.apply([], matrix);
}
