import * as faceDetection from "@tensorflow-models/face-detection";
import drawFaces from "./tensorflowUtil";
// Register WebGL backend.
import "@tensorflow/tfjs-backend-webgl";

let instance;

class FaceDetector {
  constructor() {
    if (instance) {
      throw new Error("You can only create one instance!");
    }
    this.detector = null;
    this.detectorInterval = null;
    instance = this;
  }

  getInstance() {
    return this;
  }

  setDetector(detector) {
    this.detector = detector;
  }

  /**
   * load model and create detector
   *
   * @param onDetectorLoaded callback when detector loaded
   */
  async createDetector(onDetectorLoaded) {
    try {
      onDetectorLoaded(false);
      // console.log('detector loading')
      // check current detector
      if (this.detector !== null) {
        // console.log('detector exist')
        onDetectorLoaded(true);
        return this.detector;
      }
      const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
      const detectorConfig = {
        runtime: "tfjs",
        maxFaces: 5,
      };

      const faceDetector = await faceDetection.createDetector(
        model,
        detectorConfig
      );
      this.setDetector(faceDetector);
      // console.log('detector loaded')
      onDetectorLoaded(true);
      return this.detector;
    } catch (error) {
      console.log("Error: ", error);
      throw new Error(error);
    }
  }

  /**
   * start face detector
   *
   * @param video input video
   * @param canvas canvas component for draw detection
   * @param onDetectorLoaded callback when detector loaded
   * @param onFaceDetected callback when face detected
   * @param onMultiFaceDetected callback when face detected > 1
   * @param onStarted callback when detector started
   */
  async startFaceDetector(
    video,
    canvas,
    {
      onStarted = () => {},
      onDetectorLoaded,
      onFaceDetected,
      onMultiFaceDetected,
    }
  ) {
    if (this.detector == null) {
      await this.createDetector((loaded) => onDetectorLoaded(loaded));
    }
    // console.log('detector started')
    if (this.detectorInterval === null) {
      this.detectorInterval = setInterval(() => {
        this.detect(video, canvas, onFaceDetected, onMultiFaceDetected);
      }, 100);
    }
    onStarted();
  }

  /**
   * detect faces predictions
   *
   * @param input input source detection
   * @param canvas canvas component for draw detection
   * @param onFaceDetected callback when face detected
   * @param onMultiFaceDetected callback when face detected > 1
   */
  async detect(input, canvas, onFaceDetected, onMultiFaceDetected) {
    const estimationConfig = { flipHorizontal: false };
    const faces = await this.detector.estimateFaces(input, estimationConfig);
    const ctx = canvas.getContext("2d");
    // console.log(faces)

    if (faces.length !== 0) onFaceDetected(true);
    else onFaceDetected(false);

    if (faces.length > 1) onMultiFaceDetected(true);
    else onMultiFaceDetected(false);

    requestAnimationFrame(() => drawFaces(faces, ctx));
  }

  /**
   * detect single image
   *
   * @param input input source detection (image)
   * @param canvas canvas component for draw detection
   * @param onDetectorLoaded callback when detector loaded
   */
  async detectImage(input, canvas, { onDetectorLoaded = () => {} } = {}) {
    if (this.detector == null) {
      await this.createDetector(onDetectorLoaded);
    }
    const estimationConfig = { flipHorizontal: false };
    const faces = await this.detector.estimateFaces(input, estimationConfig);
    const ctx = canvas.getContext("2d");
    drawFaces(faces, ctx, {
      withClear: false,
      withKeypoint: false,
      boxWidth: 2,
    });
    return faces;
  }

  /**
   * stop face detector
   *
   */
  stopFaceDetector(onStop = () => {}) {
    clearInterval(this.detectorInterval);
    this.detectorInterval = null;
    onStop();
    // console.log('detector stopped')
  }
}

const faceDetector = new FaceDetector();

export default faceDetector;
