/**
 *  draw face detections canvas
 *
 * @param faces detections ace
 * @param ctx canvas from canvas.getContext('2d')
 * @param withClear draw with clear first (used in realtime detection)
 * @param withKeypoint used keypoint default is true
 * @param pointColor color of keypoint
 * @param boxColor color of border box detection
 * @param boxWidth lineWidth of border box detection
 */
const drawFaceMarkers = (
  faces,
  ctx,
  {
    withClear = true,
    withKeypoint = true,
    pointColor = "aquamarine",
    boxColor = "tomato",
    boxWidth = 3,
  } = {}
) => {
  try {
    if (withClear) {
      // clear the canvas after every drawing
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
    faces.forEach((face) => {
      // do not draw if there is no face
      if (!face) return;
      const kp = face.keypoints;
      // do not draw if there is no keypoints
      if (!kp) return;
      const keypoints = kp.map((keypoint) => [keypoint.x, keypoint.y]);

      // draw box
      drawPath(
        ctx,
        [
          [face.box.xMin, face.box.yMin],
          [face.box.xMax, face.box.yMin],
          [face.box.xMax, face.box.yMax],
          [face.box.xMin, face.box.yMax],
        ],
        true,
        boxColor,
        boxWidth
      );

      if (withKeypoint) {
        // draw keypoint
        for (let i = 0; i < keypoints.length; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];

          ctx.beginPath();
          ctx.arc(x, y, 1, 0, 3 * Math.PI);
          ctx.fillStyle = pointColor;
          ctx.fill();
        }
      }
    });
  } catch (error) {
    console.log("@drawFaceMarkers error: ", error);
    throw new Error(error);
  }
};

function drawPath(ctx, points, closePath, color, lineWidth) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke(region);
}

export default drawFaceMarkers;
