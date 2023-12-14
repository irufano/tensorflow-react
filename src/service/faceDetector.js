import * as tf from '@tensorflow/tfjs-core'
import drawFaceMarkers from './markerUtil'
import { createFaceDetector, fdErrorKey } from './detector'

// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl'

let instance

class FaceDetector {
  constructor() {
    if (instance) {
      throw new Error(`${fdErrorKey} You can only create one instance!`)
    }
    this.detector = null
    this.detectorInterval = null
    instance = this
  }

  getInstance() {
    return this
  }

  setDetector(detector) {
    this.detector = detector
  }

  /**
   * load model and create detector
   *
   * @param onDetectorLoaded callback when detector loaded
   */
  async createDetector(onDetectorLoaded) {
    try {
      await tf.setBackend('webgl')
      onDetectorLoaded(false)
      // console.log('detector loading')
      // check current detector
      if (this.detector !== null) {
        // console.log('detector exist')
        onDetectorLoaded(true)
        return this.detector
      }
      // const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
      // const detectorConfig = {
      //   runtime: "tfjs",
      //   maxFaces: 5,
      //   modelType: "short",
      // };

      // const faceDetector = await faceDetection.createDetector(
      //   model,
      //   detectorConfig
      // );
      const faceDetector = await createFaceDetector({
        maxFaces: 5,
        minScoreThresh: 0.6,
      })
      this.setDetector(faceDetector)
      // console.log('detector loaded')
      onDetectorLoaded(true)
      return this.detector
    } catch (error) {
      const err = new Error(error)
      err.message = `${fdErrorKey} @createDetector error: ${err.stack}`
      throw err
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
    { onStarted = () => {}, onDetectorLoaded, onFaceDetected, onMultiFaceDetected },
  ) {
    try {
      if (this.detector == null) {
        await this.createDetector((loaded) => onDetectorLoaded(loaded))
      }

      // Add timeout to handle webgl error: Requested texture size [0x0] is invalid
      setTimeout(() => {
        if (this.detectorInterval === null) {
          this.detectorInterval = setInterval(async () => {
            await this.detect(video, canvas, onFaceDetected, onMultiFaceDetected)
          }, 100)
          onStarted()
        }
      }, 1000)
    } catch (error) {
      const err = new Error(error)
      err.message = `${fdErrorKey} @startFaceDetector error: ${err.stack}`
      throw err
    }
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
    try {
      const estimationConfig = { flipHorizontal: false }
      const faces = await this.detector.estimateFaces(input, estimationConfig)
      const ctx = canvas.getContext('2d')
      // console.log(faces)

      if (faces.length !== 0) onFaceDetected(true)
      else onFaceDetected(false)

      if (faces.length > 1) onMultiFaceDetected(true)
      else onMultiFaceDetected(false)

      requestAnimationFrame(() => drawFaceMarkers(faces, ctx))
    } catch (error) {
      const err = new Error(error)
      err.message = `${fdErrorKey} @detect error: ${err.stack}`
      throw err
    }
  }

  /**
   * detect single image
   *
   * @param input input source detection (image)
   * @param canvas canvas component for draw detection
   * @param onDetectorLoaded callback when detector loaded
   */
  async detectImage(input, canvas, { onDetectorLoaded = () => {} } = {}) {
    try {
      if (this.detector == null) {
        await this.createDetector(onDetectorLoaded)
      }
      const estimationConfig = { flipHorizontal: false }
      const faces = await this.detector.estimateFaces(input, estimationConfig)
      const ctx = canvas.getContext('2d')
      drawFaceMarkers(faces, ctx, {
        withClear: false,
        withKeypoint: false,
        boxWidth: 2,
      })
      return faces
    } catch (error) {
      const err = new Error(error)
      err.message = `${fdErrorKey} @detectImage error: ${err.stack}`
      throw err
    }
  }

  /**
   * stop face detector
   *
   */
  stopFaceDetector(onStop = () => {}) {
    try {
      clearInterval(this.detectorInterval)
      this.detectorInterval = null
      onStop()
      // console.log('detector stopped')
    } catch (error) {
      const err = new Error(error)
      err.message = `${fdErrorKey} @stopFaceDetector error: ${err.stack}`
      throw err
    }
  }
}

const faceDetector = new FaceDetector()

export default faceDetector
