import React, { useEffect, useRef, useState } from "react";
import faceDetector, { FD_ERROR_KEY } from "./service/faceDetector";

const videoResolution = {
  width: 640,
  height: 480,
};

function App() {
  const canvasRef = useRef(null);
  const webcamRef = useRef(null);
  const [counter, setCounter] = useState(0);
  const [onFrameState, serOnFrameState] = useState("");
  const [multiFaceState, setMultiFaceState] = useState("");

  let number = 0;
  let loadTimeInterval = null;

  window.addEventListener("error", function (event) {
    if (event?.message?.includes(FD_ERROR_KEY)) {
      console.log("ERROR FD: ", event);
    }
  });

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

  const startFaceDetection = async (video, stream) => {
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
          if (isOnFrame) {
            serOnFrameState("In frame");
          } else {
            serOnFrameState("Out of frame");
          }
        },
        onMultiFaceDetected: (isMultiFace) => {
          if (isMultiFace) {
            setMultiFaceState("Multi face");
          } else {
            setMultiFaceState("One face");
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
    console.log("HERE");
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
        style={{
          position: "absolute",
          color: "red",
          objectFit: "fill",
          height: `${videoResolution.height}`,
          width: `${videoResolution.width}`,
        }}
        autoPlay
      />
      <canvas
        ref={canvasRef}
        className="output_canvas"
        width={videoResolution.width}
        height={videoResolution.height}
        style={{ position: "absolute" }}
      ></canvas>
      <div
        style={{
          position: "absolute",
          padding: "8px",
          backgroundColor: "purple",
        }}
      >
        <div>
          <p
            style={{
              color: "white",
              fontSize: "12px",
              fontWeight: "bolder",
              margin: "0px 0px 6px",
            }}
          >
            Load time
          </p>
          <p
            style={{ color: "white", fontSize: "12px", margin: "0px 0px 10px" }}
          >
            {counter} s
          </p>
        </div>

        <div>
          <p
            style={{
              color: "white",
              fontSize: "12px",
              fontWeight: "bolder",
              margin: "0px",
            }}
          >
            STATUS
          </p>
          <p
            style={{ color: "white", fontSize: "12px", margin: "6px 0px 0px" }}
          >
            {onFrameState}
          </p>
          <p
            style={{ color: "white", fontSize: "12px", margin: "0px 0px 20px" }}
          >
            {multiFaceState}
          </p>
        </div>

        <button onClick={handleScreenshot}>Snapshot</button>
      </div>
    </div>
  );
}

export default App;
