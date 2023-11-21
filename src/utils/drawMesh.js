import { TRIANGULATION } from "./triangulation";

export const NUM_KEYPOINTS = 468;
export const NUM_IRIS_KEYPOINTS = 5;

export const drawMesh = (prediction, ctx) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); //clear the canvas after every drawing
  if (!prediction) return; // do not draw if there is no mesh
  const kp = prediction.keypoints;
  if (!kp) return; // do not draw if there is no keypoints
  const keypoints = kp.map((keypoint) => [keypoint.x, keypoint.y]);
  console.log(keypoints);

  const box = prediction.box;
  drawPath(
    ctx,
    [
      [box.xMin, box.yMin],
      [box.xMax, box.yMin],
      [box.xMax, box.yMax],
      [box.xMin, box.yMax],
    ],
    true
  );

  for (let i = 0; i < TRIANGULATION.length / 3; i++) {
    const points = [
      TRIANGULATION[i * 3],
      TRIANGULATION[i * 3 + 1],
      TRIANGULATION[i * 3 + 2],
    ].map((index) => keypoints[index]);

    drawPath(ctx, points, true);
  }

  for (let i = 0; i < keypoints.length; i++) {
    const x = keypoints[i][0];
    const y = keypoints[i][1];

    ctx.beginPath();
    ctx.arc(x, y, 2 /* radius */, 0, 2 * Math.PI);
    ctx.fillStyle = "aqua";
    ctx.fill();
  }
};

export const multiDrawMesh = (faces, ctx) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); //clear the canvas after every drawing
  faces.forEach((prediction) => {
    if (!prediction) return; // do not draw if there is no mesh
    const kp = prediction.keypoints;
    if (!kp) return; // do not draw if there is no keypoints
    const keypoints = kp.map((keypoint) => [keypoint.x, keypoint.y]);
    console.log(keypoints);

    const box = prediction.box;
    drawPath(
      ctx,
      [
        [box.xMin, box.yMin],
        [box.xMax, box.yMin],
        [box.xMax, box.yMax],
        [box.xMin, box.yMax],
      ],
      true
    );

    for (let i = 0; i < TRIANGULATION.length / 3; i++) {
      const points = [
        TRIANGULATION[i * 3],
        TRIANGULATION[i * 3 + 1],
        TRIANGULATION[i * 3 + 2],
      ].map((index) => keypoints[index]);

      drawPath(ctx, points, true);
    }

    for (let i = 0; i < keypoints.length; i++) {
      const x = keypoints[i][0];
      const y = keypoints[i][1];

      ctx.beginPath();
      ctx.arc(x, y, 2 /* radius */, 0, 2 * Math.PI);
      ctx.fillStyle = "aqua";
      ctx.fill();
    }
  });
};

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.strokeStyle = "tomato";
  ctx.lineWidth = 2;
  ctx.stroke(region);
}
