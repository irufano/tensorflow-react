import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as faceDetection from "@tensorflow-models/face-detection";
import { drawMesh } from "./drawMesh";
import { drawFace } from "./drawFace";

export const runMeshDetector = async (
  video,
  canvas,
  onModelLoaded,
  onFaceDetected,
  onMoreThanOneFace
) => {
  onModelLoaded(false);
  const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
  const detectorConfig = {
    runtime: "tfjs",
    maxFaces: 10,
    refineLandmarks: true,
  };
  const detector = await faceLandmarksDetection.createDetector(
    model,
    detectorConfig
  );
  const detect = async (net) => {
    const estimationConfig = { flipHorizontal: false };
    const faces = await net.estimateFaces(video, estimationConfig);
    onModelLoaded(true);
    const ctx = canvas.getContext("2d");
    // console.log(faces);
    if (faces.length !== 0) {
      onFaceDetected(true);
    } else {
      onFaceDetected(false);
    }

    if (faces.length > 1) {
      onMoreThanOneFace(true);
    } else {
      onMoreThanOneFace(false);
    }

    requestAnimationFrame(() => drawMesh(faces[0], ctx));

    detect(detector); //rerun the detect function after estimating
  };
  detect(detector); //first run of the detect function
};

export const runFaceDetector = async (
  video,
  canvas,
  onModelLoaded,
  onFaceDetected
) => {
  onModelLoaded(false);
  const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
  const detectorConfig = {
    runtime: "tfjs",
  };
  const detector = await faceDetection.createDetector(model, detectorConfig);
  const detect = async (net) => {
    const estimationConfig = { flipHorizontal: false };
    const faces = await net.estimateFaces(video, estimationConfig);
    onModelLoaded(true);
    const ctx = canvas.getContext("2d");
    console.log(faces);
    if (faces.length !== 0) {
      onFaceDetected(true);
    } else {
      onFaceDetected(false);
    }
    requestAnimationFrame(() => drawFace(faces[0], ctx));

    detect(detector); //rerun the detect function after estimating
  };
  detect(detector);
};
