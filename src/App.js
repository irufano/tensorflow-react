import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
// Register WebGL backend.
import "@tensorflow/tfjs-backend-webgl";
import "@mediapipe/face_mesh";
import { runMeshDetector } from "./utils/detector";

const inputResolution = {
  width: 1080,
  height: 900,
};

const videoConstraints = {
  width: inputResolution.width,
  height: inputResolution.height,
  facingMode: "user",
};

function App() {
  const canvasRef = useRef(null);
  const webcamRef = useRef(null);

  const [modelLoading, setModelLoading] = useState(false);
  const [cameraLoaded, setCameraLoaded] = useState(false);
  const [inFrame, setInframe] = useState(false);
  const [moreThanOneFace, setMoreThanOneFace] = useState(false);

  const onModelLoaded = (modelLoaded) => {
    if (modelLoaded) {
      setModelLoading(false);
    }
  };

  const onInframeChange = (inFrame) => {
    setInframe(inFrame);
  };

  const onMoreThanOneFace = (isMore) => {
    setMoreThanOneFace(isMore);
  };

  const handleVideoLoad = (videoNode) => {
    const video = videoNode.target;
    if (video.readyState !== 4) return;
    if (cameraLoaded) return;
    setModelLoading(true);
    runMeshDetector(
      video,
      canvasRef.current,
      onModelLoaded,
      onInframeChange,
      onMoreThanOneFace
    ); //running detection on video
    setCameraLoaded(true);
  };

  return (
    <div>
      <div>
        <Webcam
          ref={webcamRef}
          width={inputResolution.width}
          height={inputResolution.height}
          style={{ position: "absolute" }}
          videoConstraints={videoConstraints}
          onLoadedData={handleVideoLoad}
        />
        <canvas
          ref={canvasRef}
          width={inputResolution.width}
          height={inputResolution.height}
          style={{ position: "absolute" }}
        />
        <div
          style={{
            color: "aquamarine",
            textShadow: "0 0 2px #000",
          }}
        >
          {cameraLoaded ? <></> : <h1>Loading camera...</h1>}
        </div>
      </div>

      <div
        style={{
          padding: `${inputResolution.height}px 20px 20px 20px`,
        }}
      >
        <div
          style={{
            color: "aquamarine",
            textShadow: "0 0 2px #000",
          }}
        >
          {modelLoading ? <h1>Loading model...</h1> : <></>}
        </div>
        <h4>STATUS:</h4>
        <h1 style={{ color: inFrame ? "DarkTurquoise" : "MediumSlateBlue" }}>
          {inFrame ? "In frame" : "Out of frame"}
        </h1>

        <h4>TOTAL:</h4>
        <h1
          style={{
            color: moreThanOneFace ? "MediumSlateBlue" : "DarkTurquoise",
          }}
        >
          {moreThanOneFace ? "Face > 1" : "1 face"}
        </h1>
      </div>
    </div>
  );
}

export default App;
