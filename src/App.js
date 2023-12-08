import React, { useEffect, useRef, useState } from "react";
import faceDetector from "./service/faceDetector";

const videoResolution = {
  width: 640,
  height: 480,
};

function App() {
  const canvasRef = useRef(null);
  const webcamRef = useRef(null);
  const [counter, setCounter] = useState(0);
  let number = 0;
  let loadTimeInterval = null;

  const startCamera = async () => {
    const constraint = {
      video: {
        width: videoResolution.width,
        height: videoResolution.height,
      },
      audio: false,
    };
    await navigator.mediaDevices
      .getUserMedia(constraint)
      .then(async (stream) => {
        const camera = webcamRef?.current;
        camera.srcObject = stream;
        if (camera && stream.active) {
          startFaceDetection(camera, stream);
        }
      });
  };

  const startFaceDetection = (video, stream) => {
    faceDetector
      .startFaceDetector(video, canvasRef.current, {
        onDetectorLoaded: (detectorLoaded) => {
          if (!detectorLoaded) {
            console.log("detector loading...");
            if (loadTimeInterval != null) {
              clearInterval(loadTimeInterval);
            }
            loadTimeInterval = setInterval(() => {
              number++;
              setCounter(number);
              // console.log(number);
            }, 1000);
          } else {
            console.log("detector loaded");
            clearInterval(loadTimeInterval);
          }
        },
        onStarted: () => {
          console.log("detector started");
          // in some device if refresh camera not stream anymore then stop detector
          if (!stream.active) {
            stopDetector();
          }
        },
        onFaceDetected: (isOnFrame) => {
          if (!isOnFrame) {
            console.log("Out of frame!");
          }
        },
        onMultiFaceDetected: (isMultiFace) => {
          if (isMultiFace) {
            console.log("Only one face allowed");
          }
        },
      })
      .catch((err) => {
        console.log("detectorError : ", err);
      });
  };

  const stopDetector = () => {
    faceDetector.stopFaceDetector(() => {
      console.log("detector stopped");
    });
  };

  const handleScreenshot = async () => {
    try {
      const videoElement = document.getElementById("web-cam");
      const canvasElement = document.createElement("canvas");

      canvasElement.width = videoElement?.videoWidth;
      canvasElement.height = videoElement?.videoHeight;

      console.log(canvasElement);

      const canvasContext = canvasElement.getContext("2d");
      canvasContext.drawImage(
        videoElement,
        0,
        0,
        canvasElement?.width,
        canvasElement?.height
      );

      await faceDetector.detectImage(canvasElement, canvasElement);

      const dataUrl = canvasElement.toDataURL();

      console.log(dataUrl);
    } catch (error) {
      console.log("ERROR SS: ", error);
    }
  };

  useEffect(() => {
    startCamera();
    // eslint-disable-next-line
  }, []);

  return (
    <div>
      <video
        id="web-cam"
        ref={webcamRef}
        width={videoResolution.width}
        height={videoResolution.height}
        style={{ position: "absolute", color: "red", objectFit: "fill" }}
        autoPlay
      />
      <canvas
        ref={canvasRef}
        className="output_canvas"
        width={videoResolution.width}
        height={videoResolution.height}
        style={{ position: "absolute" }}
      ></canvas>
      <div style={{ position: "absolute", padding: "8px" }}>
        <h3 style={{ color: "white" }}>Load time: {counter} s</h3>
        <button onClick={handleScreenshot}>Snapshot</button>
      </div>
    </div>
  );
}

export default App;
