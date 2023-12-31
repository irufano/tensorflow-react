export const drawFace = (prediction, ctx) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); //clear the canvas after every drawing
  if (!prediction) return; // do not draw if there is no mesh
  const keyPoints = prediction.keypoints;
  if (!keyPoints) return; // do not draw if there is no keypoints
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
  for (let keyPoint of keyPoints) {
    ctx.beginPath();
    ctx.arc(keyPoint.x, keyPoint.y, 2, 0, 3 * Math.PI);
    ctx.fillStyle = "aquamarine";
    ctx.fill();
  }
};

export const multiDrawFace = (faces, ctx) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); //clear the canvas after every drawing
  faces.forEach((prediction) => {
    if (!prediction) return; // do not draw if there is no mesh
    const kp = prediction.keypoints;
    if (!kp) return; // do not draw if there is no keypoints
    const keypoints = kp.map((keypoint) => [keypoint.x, keypoint.y]);

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
    for (let i = 0; i < keypoints.length; i++) {
      const x = keypoints[i][0];
      const y = keypoints[i][1];
  
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 3 * Math.PI);
      ctx.fillStyle = "aquamarine";
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
  ctx.lineWidth = 4;
  ctx.stroke(region);
}
