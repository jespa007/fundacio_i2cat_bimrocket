/**
 * GeometryUtils.js
 *
 * @author realor
 */

import * as THREE from "three";
import earcut from "earcut";

class GeometryUtils
{
  static PRECISION = 0.00001;
  static _vector1 = new THREE.Vector3();
  static _vector2 = new THREE.Vector3();
  static _vector3 = new THREE.Vector3();
  static _vector4 = new THREE.Vector3();
  static _vector5 = new THREE.Vector3();
  static _plane1 = new THREE.Plane();

  static isPointOnSegment(point, pointA, pointB, distance = 0.0001)
  {
    const projectedPoint = this._vector1;

    if (this.projectPointOnSegment(point, pointA, pointB, projectedPoint))
    {
      return projectedPoint.distanceToSquared(point) < distance * distance;
    }
    return false;
  }

  static projectPointOnSegment(point, pointA, pointB, projectedPoint)
  {
    const vAB = this._vector2;
    const vAP = this._vector3;
    const vProjAB = this._vector4;

    vAB.subVectors(pointB, pointA);
    vAP.subVectors(point, pointA);

    const denominator = vAB.lengthSq();

    if (denominator === 0) return null;

		const scalar = vAB.dot(vAP) / denominator;

    if (scalar >= 0 && scalar <= 1)
    {
		  vProjAB.copy(vAB).multiplyScalar(scalar);

      if (!(projectedPoint instanceof THREE.Vector3))
      {
        projectedPoint = new THREE.Vector3();
      }
      projectedPoint.copy(pointA).add(vProjAB);

      return projectedPoint;
    }
    return null;
  }

  static intersectLines(line1, line2, position1, position2)
  {
    const vector1 = GeometryUtils._vector1;
    const vector2 = GeometryUtils._vector2;
    const normal = GeometryUtils._vector5;
    const plane = GeometryUtils._plane1;
    position1 = position1 || GeometryUtils._vector3;
    position2 = position2 || GeometryUtils._vector4;

    vector1.subVectors(line1.end, line1.start).normalize();
    vector2.subVectors(line2.end, line2.start).normalize();

    if (Math.abs(vector1.dot(vector2)) < 0.9999) // are not parallel
    {
      normal.copy(vector1).cross(vector2).normalize();

      vector1.cross(normal).normalize();
      plane.setFromNormalAndCoplanarPoint(vector1, line1.start);
      if (plane.intersectLine(line2, position2))
      {
        vector2.cross(normal).normalize();
        plane.setFromNormalAndCoplanarPoint(vector2, line2.start);
        if (plane.intersectLine(line1, position1))
        {
          return position1.distanceTo(position2);
        }
      }
    }
    return -1;
  }

  static centroid(vertexPositions, accessFn, centroid)
  {
    if (!(centroid instanceof THREE.Vector3)) centroid = new THREE.Vector3();
    else centroid.set(0, 0, 0);

    const count = vertexPositions.length;
    let point;
    for (let i = 0; i < count; i++)
    {
      if (accessFn)
      {
        point = accessFn(vertexPositions[i]);
      }
      else
      {
        point = vertexPositions[i];
      }
      centroid.x += point.x;
      centroid.y += point.y;
      centroid.z += point.z;
    }
    centroid.x /= count;
    centroid.y /= count;
    centroid.z /= count;

    return centroid;
  }

  static calculateNormal(vertexPositions, accessFn, normal)
  {
    if (!(normal instanceof THREE.Vector3)) normal = new THREE.Vector3();
    else normal.set(0, 0, 0);

    // Newell's method
    const count = vertexPositions.length;
    let pi, pj;
    for (let i = 0; i < count; i++)
    {
      let j = (i + 1) % count;
      if (accessFn)
      {
        pi = accessFn(vertexPositions[i]);
        pj = accessFn(vertexPositions[j]);
      }
      else
      {
        pi = vertexPositions[i];
        pj = vertexPositions[j];
      }
      normal.x += (pi.y - pj.y) * (pi.z + pj.z);
      normal.y += (pi.z - pj.z) * (pi.x + pj.x);
      normal.z += (pi.x - pj.x) * (pi.y + pj.y);
    }
    normal.normalize();
    return normal;
  }

  /**
   * Converts a local normal to WCS
   *
   * @param {THREE.Vector3} normal - vector to convert to WCS
   * @param {THREE.Matrix4} matrixWorld - the matrix where normal is referenced
   * @param {THREE.Vector3} worldNormal - the output normal
   * @returns {THREE.Vector3} the worldNormal
   */
  static getWorldNormal(normal, matrixWorld, worldNormal)
  {
    if (!(worldNormal instanceof THREE.Vector3))
    {
      worldNormal = new THREE.Vector3();
    }

    const v0 = this._vector1;
    const v1 = this._vector2;

    v0.setFromMatrixPosition(matrixWorld);
    v1.copy(normal).applyMatrix4(matrixWorld);
    v1.sub(v0).normalize();
    worldNormal.copy(v1);

    return worldNormal;
  }

  /**
   * Returns the center of a circle that passes through the 3 given points.
   *
   * @param {THREE.Vector2} point1
   * @param {THREE.Vector2} point2
   * @param {THREE.Vector2} point3
   * @returns {THREE.Vector2}
   */
  static getCircleCenter(point1, point2, point3)
  {
    const x1 = point1.x;
    const y1 = point1.y;
    const x2 = point2.x;
    const y2 = point2.y;
    const x3 = point3.x;
    const y3 = point3.y;

    const x1_2 = x1 * x1;
    const x2_2 = x2 * x2;
    const x3_2 = x3 * x3;
    const y1_2 = y1 * y1;
    const y2_2 = y2 * y2;
    const y3_2 = y3 * y3;

    const t1 = (x2_2 + y2_2 - x3_2 - y3_2);
    const t2 = (x1_2 + y1_2 - x2_2 - y2_2);

    const xc = ((y2 - y1) * t1 - (y3 - y2) * t2) /
      (2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)));

    const yc = ((x2 - x1) * t1 - (x3 - x2) * t2) /
      (2 * (y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)));

    if (!isFinite(xc) || !isFinite(yc)) return null;

    return new THREE.Vector2(xc, yc);
  }

  /**
   * Returns the offset vector that should be subtracted from the given point
   * to avoid lost of precision when its coordinates are represented with float32.
   *
   * @param {THREE.Vector2 | THREE.Vector3} point
   * @returns {THREE.Vector3} the offsetVector
   */
  static getOffsetVectorForFloat32(point)
  {
    const maxCoord = 10000;
    let offsetVector = null;

    let xOverflow = Math.abs(point.x) > maxCoord;
    let yOverflow = Math.abs(point.y) > maxCoord;
    let zOverflow = point.isVector3 && Math.abs(point.z) > maxCoord;

    if (xOverflow || yOverflow || zOverflow)
    {
      offsetVector = new THREE.Vector3();
      if (xOverflow) offsetVector.x = point.x;
      if (yOverflow) offsetVector.y = point.y;
      if (zOverflow) offsetVector.z = point.z;
    }
    return offsetVector;
  }

  /**
   * Subtracts offsetVector from the points of the given rings.
   *
   * @param {Vector3} offsetVector
   * @param {Vector2[] | Vector3[]} rings
   */
  static offsetRings(offsetVector, ...rings)
  {
    for (let ring of rings)
    {
      for (let i = 0; i < ring.length - 1; i++)
      {
        ring[i].sub(offsetVector);
      }
      if (ring[0] !== ring[ring.length - 1])
      {
        ring[ring.length - 1].sub(offsetVector);
      }
    }
  }

  /**
   * Clones an array of Vector2 or Vector3
   *
   * @param {Vector2[] | Vector3[]} ring - the array of vectors
   * @returns {Vector2[] | Vector3} the cloned ring
   */
  static cloneRing(ring)
  {
    if (!(ring instanceof Array)) return ring;

    const clonedRing = [];

    for (let vector of ring)
    {
      clonedRing.push(vector.clone());
    }
    return clonedRing;
  }

  /**
   * Add nextPoints to points without repeating vertices
   *
   * @param {Vector2[] | Vector3[]} points - the array of vectors
   * @param {Vector2[] | Vector3[]} nextPoints - the array of vectors
   */
  static joinPointArrays(points, nextPoints)
  {
    if (nextPoints.length === 0)
    {
      // nothing to do
    }
    else if (points.length === 0)
    {
      points.push(...nextPoints);
    }
    else
    {
      const lastPoint = points[points.length - 1];
      const nextPoint = nextPoints[0];
      let start = lastPoint.equals(nextPoint) ? 1 : 0;
      for (let i = start; i < nextPoints.length; i++)
      {
        points.push(nextPoints[i]);
      }
    }
  }

  /* triangulate a 3D face */
  static triangulateFace(vertices, holes, normal)
  {
    const vx = GeometryUtils._vector1;
    const vy = GeometryUtils._vector2;
    const vz = GeometryUtils._vector3;

    if (normal instanceof THREE.Vector3)
    {
      vz.copy(normal);
    }
    else
    {
      GeometryUtils.calculateNormal(vertices, undefined, vz);
    }
    const v0 = vertices[0];
    const v1 = vertices[1];
    vx.subVectors(v1, v0).normalize();
    vy.crossVectors(vz, vx);

    const matrix = new THREE.Matrix4();
    matrix.set(vx.x, vy.x, vz.x, v0.x,
             vx.y, vy.y, vz.y, v0.y,
             vx.z, vy.z, vz.z, v0.z,
             0, 0, 0, 1).invert();

    const projectVertices = (vertices) =>
    {
      let projectedVertices = [];
      for (let vertex of vertices)
      {
        let point = new THREE.Vector3();
        point.copy(vertex);
        projectedVertices.push(point.applyMatrix4(matrix));
      }
      return projectedVertices;
    };

    let projectedVertices = projectVertices(vertices);
    let projectedHoles = [];
    for (let hole of holes)
    {
      projectedHoles.push(projectVertices(hole));
    }

    return THREE.ShapeUtils.triangulateShape(projectedVertices, projectedHoles);
  }

  //---------
  // Cylindrical/Revolution Surface Helpers

  static buildBoundaryGroups(boundVertices, boundIndices) {
    const maxGroup = Math.max(...boundIndices);

    const groups = [];
    for (let i = 0; i <= maxGroup; i++) {
        groups.push([]);
    }

    for (let i = 0; i < boundIndices.length; i++) {
        const g = boundIndices[i];
        groups[g].push(boundVertices[i]);
    }

    return groups; 
}

/* generate points/triangulate 3D revolution face with trimming */
static buildSurfaceOfRevolutionWithoutTrimming(transform,boundVertices, boundIndices, profile3d, startDeg, endDeg,  circleSegments) {

    // Extract local basis and origin from the transform

    const inverseTransform = transform.clone().invert();
    
    const xAxis = new THREE.Vector3();
    const yAxis = new THREE.Vector3();
    const zAxis = new THREE.Vector3();
    transform.extractBasis(xAxis, yAxis, zAxis);

    xAxis.normalize(); yAxis.normalize(); zAxis.normalize();
    const origin = new THREE.Vector3().setFromMatrixPosition(transform);

    const startRad = THREE.MathUtils.degToRad(startDeg);
    const endRad   = THREE.MathUtils.degToRad(endDeg);
    const span     = endRad - startRad;
    const step     = span / (circleSegments - 1);

    const rel = new THREE.Vector3();

    // Precompute profile projected to local frame
    const localProfile = profile3d.map(p => {
      rel.set(0,0,0);
      rel.subVectors(p, origin);

      const dx = rel.dot(xAxis);
      const dy = rel.dot(yAxis);
      const dz = rel.dot(zAxis);
      return { 
        dd: Math.hypot(dx, dy), dz 
      };
    });

    const ringSize = localProfile.length;
    const vertices = [];
    const triangles = [];

    for(let r=0; r < circleSegments; r++){
      const ang = startRad + r*step;
      const s = Math.sin(ang);
      const c = Math.cos(ang);

      for(let i=0; i < ringSize; i++){

        const {dd,dz} = localProfile[i];

        vertices.push(
          new THREE.Vector3()
            .addScaledVector(xAxis,s*dd)
            .addScaledVector(yAxis,c*dd)
            .addScaledVector(zAxis,dz)
            .add(origin)      
        );
      }

    }

   // Triangulate between rings
    const idx = (r, s) => r * ringSize + s;
    for (let r = 0; r < circleSegments - 1; r++) {
      for (let i = 0; i < ringSize - 1; i++) {
        const a = idx(r, i);          // 0
        const b = idx(r, i + 1);      // 1
        const c = idx(r + 1, i);      // 2
        const d = idx(r + 1, i + 1);  // 3
        //faces.push([[a, b, c], [c, b, d]]);

        const dd_i   = localProfile[i].dd;     // from projection
        const dd_ip1 = localProfile[i+1].dd;   // from projection

        // CASE 1
        // If profile hits the axis here,
        // use triangle fans, not quads
        if (dd_i < GeometryUtils.PRECISION) {
            triangles.push([a, c, d]); // triangle fan
            continue;
        }

        if (dd_ip1 < GeometryUtils.PRECISION) {
            triangles.push([a, b, c]); // other triangle fan
            continue;
        }

        // CASE 2
        // Otherwise it's a regular quad -> 2 triangles
        triangles.push([a, b, c]);
        triangles.push([c, b, d]);
      }
    }
    
    return [vertices,triangles];
  }
 
/*  buildSurfaceOfRevolution
 *  - Parametrization: (u = angle, v = arc-length along profile)
 *  - profile3d and boundVertices are in WORLD coordinates
 *  - `transform` is the local frame of the surface-of-revolution
 *      (origin = IfcAxis1Placement.Location,
 *       z-axis = IfcAxis1Placement.Axis,
 *       x/y are orthogonal directions)
 *  - Trims in UV space using martinez + earcut, EXCEPT when the
 *    boundary polygon corresponds to an untrimmed surface:
 *      * full param rectangle (dome, full revolution), or
 *      * only two loops at v close to 0 and v close to 1 (partial-span, like half cylinder).
 */
static buildSurfaceOfRevolution(
    transform,      // THREE.Matrix4
    boundVertices,  // THREE.Vector3[]
    boundIndices,   // (not used yet - single loop assumed)
    profile3d,      // THREE.Vector3[]
    startDeg,
    endDeg,
    circleSegments
) {
    const inverseTransform = transform.clone().invert();

    // Extract local basis (columns of the matrix)
    const xAxis = new THREE.Vector3();
    const yAxis = new THREE.Vector3();
    const zAxis = new THREE.Vector3();
    transform.extractBasis(xAxis, yAxis, zAxis);
    xAxis.normalize(); yAxis.normalize(); zAxis.normalize();

    const origin = new THREE.Vector3().setFromMatrixPosition(transform);

    const startRad = THREE.MathUtils.degToRad(startDeg);
    const endRad   = THREE.MathUtils.degToRad(endDeg);
    const span     = endRad - startRad;
    const step     = span / (circleSegments - 1);

    const rel = new THREE.Vector3();

    // ---------------------------------------------------------------------
    // 1) Profile in LOCAL frame: (dd, dz) + arc-length parameter v
    // ---------------------------------------------------------------------
    const localProfile = [];   // { dd, dz }
    const prof2D       = [];   // THREE.Vector2(dd, dz)
    const cumLen       = [];   // cumulative arc length

    let accLen = 0;
    for (let i = 0; i < profile3d.length; i++) {
        const p = profile3d[i];

        // world -> local
        rel.copy(p).sub(origin);
        const dx = rel.dot(xAxis);
        const dy = rel.dot(yAxis);
        const dz = rel.dot(zAxis);

        const dd = Math.hypot(dx, dy);

        localProfile.push({ dd, dz });
        const pt2 = new THREE.Vector2(dd, dz);
        prof2D.push(pt2);

        if (i === 0) {
            cumLen.push(0);
        } else {
            const prev = prof2D[i - 1];
            const segLen = pt2.distanceTo(prev);
            accLen += segLen;
            cumLen.push(accLen);
        }
    }

    const totalLen = accLen > 0 ? accLen : 1.0;
    const ringSize = localProfile.length;

    // normalized v for each profile sample (0..1)
    const profileV = cumLen.map(s => s / totalLen);

    // ---------------------------------------------------------------------
    // 2) Helper: project arbitrary local (dd,dz) onto profile to get v
    // ---------------------------------------------------------------------
    function projectToProfile(dd, dz) {
        const p = new THREE.Vector2(dd, dz);

        let bestSeg = 0;
        let bestT = 0;
        let bestDist2 = Infinity;

        for (let i = 0; i < prof2D.length - 1; i++) {
            const a = prof2D[i];
            const b = prof2D[i + 1];

            const abx = b.x - a.x;
            const aby = b.y - a.y;
            const apx = p.x - a.x;
            const apy = p.y - a.y;

            const abLenSq = abx * abx + aby * aby;
            let t = 0;
            if (abLenSq > 1e-16) {
                t = (apx * abx + apy * aby) / abLenSq;
                if (t < 0) t = 0;
                else if (t > 1) t = 1;
            }

            const cx = a.x + t * abx;
            const cy = a.y + t * aby;
            const dx = p.x - cx;
            const dy = p.y - cy;
            const dist2 = dx * dx + dy * dy;

            if (dist2 < bestDist2) {
                bestDist2 = dist2;
                bestSeg = i;
                bestT = t;
            }
        }

        const segStartLen = cumLen[bestSeg];
        const segLen = cumLen[bestSeg + 1] - cumLen[bestSeg];
        const s = segStartLen + bestT * segLen;
        return s / totalLen;
    }

    // ---------------------------------------------------------------------
    // 3) Sample untrimmed surface grid and assign UV = (u, v)
    // ---------------------------------------------------------------------
    const gridXYZ = []; // 3D points on surface
    const gridUV  = []; // {u, v}

    for (let r = 0; r < circleSegments; r++) {
        const ang = startRad + r * step;
        const s = Math.sin(ang);
        const c = Math.cos(ang);
        const u = ang; // parametric angle

        for (let i = 0; i < ringSize; i++) {
            const { dd, dz } = localProfile[i];
            const v = profileV[i];

            const p = new THREE.Vector3()
                .addScaledVector(xAxis, s * dd)
                .addScaledVector(yAxis, c * dd)
                .addScaledVector(zAxis, dz)
                .add(origin);

            gridXYZ.push(p);
            gridUV.push({ u, v });
        }
    }

    const idx = (r, s) => r * ringSize + s;

    // ---------------------------------------------------------------------
    // 4) Build trimming boundary in UV space from boundVertices
    // ---------------------------------------------------------------------
    function unwrapToMatch(u, refStart) {
        while (u < refStart - Math.PI) u += 2 * Math.PI;
        while (u > refStart + Math.PI) u -= 2 * Math.PI;
        return u;
    }

    const surfU0 = gridUV[0].u; // usually startRad

    const boundaryRingUV = [];

    for(let i = 0; i < boundVertices.length; i++){
      const p = boundVertices[i];

      const local = p.clone().applyMatrix4(inverseTransform);

      // angle around local Z-axis
      let u = Math.atan2(local.x, local.y);
      u = unwrapToMatch(u, surfU0);

      const dd = Math.hypot(local.x, local.y);
      const dz = local.z;
      const v  = projectToProfile(dd, dz);

      // unwrap by adding or subtracting 2pi until difference is small
      if(i > 0){
        let prevU = boundaryRingUV[i - 1][0];
        while (u - prevU > Math.PI)  u -= 2 * Math.PI;
            while (u - prevU < -Math.PI) u += 2 * Math.PI;
      }
      
      
      boundaryRingUV.push([u,v]);
    }

    // ---------------------------------------------------------
    // Normalize whole loop to fit inside [startRad, endRad]
    // while preserving continuity.
    // ---------------------------------------------------------
    let minU = Infinity;
    let maxU = -Infinity;

    for (const [u] of boundaryRingUV) {
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
    }

    // Case: loop is entirely left of domain -> shift forward
    while (maxU < startRad) {
        for (let i = 0; i < boundaryRingUV.length; i++)
            boundaryRingUV[i][0] += 2 * Math.PI;

        minU += 2 * Math.PI;
        maxU += 2 * Math.PI;
    }

    // Case: loop is entirely right of domain -> shift backward
    while (minU > endRad) {
        for (let i = 0; i < boundaryRingUV.length; i++)
            boundaryRingUV[i][0] -= 2 * Math.PI;

        minU -= 2 * Math.PI;
        maxU -= 2 * Math.PI;
    }

    // Signed area of polygon in UV
    const signedArea = (ring) => {
        let a = 0;
        for (let i = 0, n = ring.length; i < n; i++) {
            const [x0, y0] = ring[i];
            const [x1, y1] = ring[(i + 1) % n];
            a += x0 * y1 - x1 * y0;
        }
        return 0.5 * a;
    };

    // --- after computing boundaryRingUV and signedArea(boundaryRingUV) ---

    const boundaryArea = Math.abs(signedArea(boundaryRingUV));
    const rectArea     = Math.abs(span) * 1.0; // v in [0,1]
    const areaDiffRel  = rectArea > 1e-12
        ? Math.abs(boundaryArea - rectArea) / rectArea
        : 0;

    // Relaxed area threshold
    const UNTRIMMED_AREA_THRESHOLD = 0.20; // 20%

    // ---- 1) "top & bottom only" detection ----------------
    const EPS_V = 1e-3;
    let vMin =  Infinity;
    let vMax = -Infinity;
    let midCount = 0;

    for (const [, v] of boundaryRingUV) {
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
        if (v > EPS_V && v < 1 - EPS_V) {
            midCount++;
        }
    }

    const hasBottom = vMin < EPS_V;
    const hasTop    = vMax > 1 - EPS_V;
    const noInteriorV = (midCount === 0);

    const isTopBottomOnlyUntrim = hasBottom && hasTop && noInteriorV;

    // ---- 2) "rectangle-like border" detection ---------------------
    const spanAbs = Math.abs(span);
    const EPS_U   = spanAbs * 0.1; // 10% of angle span as tolerance
    const EDGE_V_TOL = 0.1;       // 10% in v

    let interiorCount = 0;
    let countBottom = 0, countTop = 0, countLeft = 0, countRight = 0;

    for (const [u, v] of boundaryRingUV) {
        const nearBottom = (v < EDGE_V_TOL);
        const nearTop    = (v > 1 - EDGE_V_TOL);
        const nearLeft   = (Math.abs(u - startRad) < EPS_U);
        const nearRight  = (Math.abs(u - endRad)   < EPS_U);

        if (nearBottom) countBottom++;
        if (nearTop)    countTop++;
        if (nearLeft)   countLeft++;
        if (nearRight)  countRight++;

        if (!nearBottom && !nearTop && !nearLeft && !nearRight) {
            interiorCount++;
        }
    }

    const totalBV = boundaryRingUV.length;
    const interiorRatio = totalBV > 0 ? interiorCount / totalBV : 1;

    // enough points on each edge?
    const hasBottomEdge = countBottom > 0;
    const hasTopEdge    = countTop    > 0;
    const hasSideEdges  = (countLeft + countRight) > 0;

    // "looks like a rectangle border" if almost all points sit near the 4 edges
    const isRectLikeBorder =
        hasBottomEdge && hasTopEdge && hasSideEdges &&
        (interiorRatio < 0.05); // at most 5% points in the interior

    // ---- 3) Full-rectangle area-based detection (from 0.4.1) ----------
    const isFullRectUntrimmed = (areaDiffRel < UNTRIMMED_AREA_THRESHOLD);

    // ---- 4) Check untrimmed --------------------------------------------
    const isUntrimmed =
        isTopBottomOnlyUntrim ||     // v close to 0 & v close to 1 only (previous half-cylinder)
        isRectLikeBorder      ||     // NEW: full rectangle border, no interior points
        isFullRectUntrimmed;         // area close to span*1 (dome & friends)

    if (isUntrimmed) {
        // Treat as untrimmed: directly triangulate the grid.
        const vertices  = gridXYZ.slice();
        const triangles = [];
        const faces = [];

        for (let r = 0; r < circleSegments - 1; r++) {
            for (let i = 0; i < ringSize - 1; i++) {
              const v00 = gridXYZ[idx(r,     i)];
              const v10 = gridXYZ[idx(r,     i + 1)];
              const v01 = gridXYZ[idx(r + 1, i)];
              const v11 = gridXYZ[idx(r + 1, i + 1)];

              faces.push([ v00, v10, v11, v01 ]);
            }
        }

        return faces;
    }


    // ---------------------------------------------------------------------
    // 5) If not untrimmed: continue with trimming pipeline
    // ---------------------------------------------------------------------
    // helper: signed area we already defined above
    if (signedArea(boundaryRingUV) < 0) {
      boundaryRingUV.reverse();
    }

    const boundaryUV = [boundaryRingUV]; // single outer loop, no holes
    const SCALE = 1e6;

    const scalePoly = (poly) =>
        poly.map(ring =>
            ring.map(([u, v]) => [Math.round(u * SCALE), Math.round(v * SCALE)])
        );

    const unscaleMulti = (multi) =>
        multi.map(poly =>
            poly.map(ring =>
                ring.map(([x, y]) => [x / SCALE, y / SCALE])
            )
        );

    const boundaryUVInt = scalePoly(boundaryUV);

    const triangulatePolygon = (poly) => {
        const cleanRing = (ring) => {
            if (
                ring.length > 1 &&
                ring[0][0] === ring[ring.length - 1][0] &&
                ring[0][1] === ring[ring.length - 1][1]
            ) {
                return ring.slice(0, -1);
            }
            return ring;
        };

        const outer = cleanRing(poly[0]);
        const holes = poly.slice(1).map(cleanRing);

        const coords = [];
        const holeIndices = [];
        let index = 0;

        outer.forEach(([x, y]) => coords.push(x, y));
        index += outer.length;

        for (const h of holes) {
            holeIndices.push(index);
            h.forEach(([x, y]) => coords.push(x, y));
            index += h.length;
        }

        return earcut(coords, holeIndices);
    };

    const uvToXYZ = (P, Auv, Buv, Cuv, Axyz, Bxyz, Cxyz) => {
        const v0 = { x: Buv.u - Auv.u, y: Buv.v - Auv.v };
        const v1 = { x: Cuv.u - Auv.u, y: Cuv.v - Auv.v };
        const v2 = { x: P[0]  - Auv.u, y: P[1]  - Auv.v };

        const d00 = v0.x * v0.x + v0.y * v0.y;
        const d01 = v0.x * v1.x + v0.y * v1.y;
        const d11 = v1.x * v1.x + v1.y * v1.y;
        const d20 = v2.x * v0.x + v2.y * v0.y;
        const d21 = v2.x * v1.x + v2.y * v1.y;

        const denom = d00 * d11 - d01 * d01;
        if (Math.abs(denom) < 1e-18) {
            return Axyz.clone(); // degenerate
        }

        const bary_v = (d11 * d20 - d01 * d21) / denom;
        const bary_w = (d00 * d21 - d01 * d20) / denom;
        const bary_u = 1 - bary_v - bary_w;

        return new THREE.Vector3(
            bary_u * Axyz.x + bary_v * Bxyz.x + bary_w * Cxyz.x,
            bary_u * Axyz.y + bary_v * Bxyz.y + bary_w * Cxyz.y,
            bary_u * Axyz.z + bary_v * Bxyz.z + bary_w * Cxyz.z
        );
    };

    const triArea = (P0, P1, P2) =>
        Math.abs(
            P0[0] * (P1[1] - P2[1]) +
            P1[0] * (P2[1] - P0[1]) +
            P2[0] * (P0[1] - P1[1])
        ) * 0.5;

    const faces = [];

    const clipTriangle = (i0, i1, i2) => {
        const Auv  = gridUV[i0], Axyz = gridXYZ[i0];
        const Buv  = gridUV[i1], Bxyz = gridXYZ[i1];
        const Cuv  = gridUV[i2], Cxyz = gridXYZ[i2];

        const triRing = [
            [Auv.u, Auv.v],
            [Buv.u, Buv.v],
            [Cuv.u, Cuv.v],
            [Auv.u, Auv.v]
        ];

        if (triArea(triRing[0], triRing[1], triRing[2]) < 1e-12) return;

        const triPolyInt = [triRing].map(ring =>
            ring.map(([u, v]) => [Math.round(u * SCALE), Math.round(v * SCALE)])
        );

        let clippedInt;
        try {
            // subject: boundaryUVInt, clipping: triPolyInt
            clippedInt = martinez.intersection(boundaryUVInt, triPolyInt);
        } catch (e) {
            console.error("martinez.intersection error:", e);
            return;
        }

        if (!clippedInt || !clippedInt.length) return;

        const clippedFloat = unscaleMulti(clippedInt);

        for (const poly of clippedFloat) {
            const triIndices = triangulatePolygon(poly);
            if (!triIndices || triIndices.length === 0) continue;

            const outer = poly[0];

            for (let k = 0; k < triIndices.length; k += 3) {
                const ia = triIndices[k];
                const ib = triIndices[k + 1];
                const ic = triIndices[k + 2];

                const Pa = outer[ia];
                const Pb = outer[ib];
                const Pc = outer[ic];
                if (!Pa || !Pb || !Pc) continue;

                const vA = uvToXYZ(Pa, Auv, Buv, Cuv, Axyz, Bxyz, Cxyz);
                const vB = uvToXYZ(Pb, Auv, Buv, Cuv, Axyz, Bxyz, Cxyz);
                const vC = uvToXYZ(Pc, Auv, Buv, Cuv, Axyz, Bxyz, Cxyz);

                //const base = vertices.length;
                //vertices.push(vA, vB, vC);
                //triangles.push([base, base + 1, base + 2]);
                faces.push([ vA, vB, vC ]);
            }
        }
    };

    for (let r = 0; r < circleSegments - 1; r++) {
        for (let i = 0; i < ringSize - 1; i++) {
            const a = idx(r,     i);
            const b = idx(r,     i + 1);
            const c = idx(r + 1, i);
            const d = idx(r + 1, i + 1);

            clipTriangle(a, b, c);
            clipTriangle(c, b, d);
        }
    }

    return faces;
}



  /*  triangulate a 3D revolution face */
  static triangulateSurfaceOfRevolution(transform, boundVertices, boundIndices, profile3d, circleSegments=16) {
    
    //console.table(vertices);

    // Extract axes + origin for projections
    const xAxis = new THREE.Vector3();
    const yAxis = new THREE.Vector3();
    const zAxis = new THREE.Vector3();
    transform.extractBasis(xAxis, yAxis, zAxis);

    xAxis.normalize(); yAxis.normalize(); zAxis.normalize();

    const origin = new THREE.Vector3().setFromMatrixPosition(transform);

    // ----- Step 1: group boundaries and centers
    const groups = GeometryUtils.buildBoundaryGroups(boundVertices,boundIndices);


    // ----- Step 1: representative (centroid) for each boundary
    const reps = [];

    for (const group of groups) {
        let c = new THREE.Vector3(0,0,0);
        for (const p of group) {
          c.add(p);
        }
        c.multiplyScalar(1 / group.length);
        reps.push(c);
    }

    // ----- Step 2: compute each angle in frame
    const angleVec = [];

    for (const p of reps) {
        const rel = new THREE.Vector3().subVectors(p, origin);
        const dx = rel.dot(xAxis);
        const dy = rel.dot(yAxis);

        let ang = Math.atan2(dx, dy) * 180 / Math.PI;
        if (ang < 0) ang += 360;
        if (ang >= 360) ang -= 360;

        angleVec.push(ang);
    }

    // ----- Step 3: unwrap angle differences
    const angleDiff = [];
    for (let i = 0; i < angleVec.length - 1; i++) {
        let diff = angleVec[i + 1] - angleVec[i];

        if (diff > 180) diff -= 360;
        else if (diff < -180) diff += 360;

        angleDiff.push(diff);
    }    

    // ----- Step 4: accumulate and find min/max
    let startDeg = angleVec[0];
    let endDeg = angleVec[0];
    let acc = angleVec[0];

    for (const d of angleDiff) {
        acc += d;
        if (acc < startDeg) startDeg = acc;
        if (acc > endDeg) endDeg = acc;
    }



   return GeometryUtils.buildSurfaceOfRevolution(
      transform
      ,boundVertices
      ,boundIndices
      ,profile3d
      ,startDeg
      ,endDeg
      ,circleSegments
    )

  }

    /* generate points/triangulate 3D cylindrical face */
  static buildCylindricalSurface(transform, startDeg, endDeg, minZ, maxZ, radius, circleSegments) {

    const xDir = GeometryUtils._vector1;
    const yDir = GeometryUtils._vector2;
    const zDir = GeometryUtils._vector3;
    const center = GeometryUtils._vector4;

    center.setFromMatrixPosition(transform);

    transform.extractBasis(xDir, yDir, zDir);
    xDir.normalize(); yDir.normalize(); zDir.normalize();

    const startRad = THREE.MathUtils.degToRad(startDeg);
    const endRad = THREE.MathUtils.degToRad(endDeg);
    const radStep = (endRad - startRad) / (circleSegments - 1);
    const zStep = (maxZ - minZ) / (circleSegments - 1);

  // --- 1. Build rings with ONLY 2 height samples (minZ, maxZ) ---
  const rings = []; // rings[r][j], j = 0(minZ), 1(maxZ)
  for (let r = 0; r < circleSegments; r++) {
    const angle = startRad + r * radStep;
    const sinA = Math.sin(angle);
    const cosA = Math.cos(angle);

    const ring = [];

    // j = 0 -> minZ
    {
      const p0 = new THREE.Vector3()
        .addScaledVector(xDir, sinA * radius)
        .addScaledVector(yDir, cosA * radius)
        .addScaledVector(zDir, minZ)
        .add(center);
      ring.push(p0);
    }

    // j = 1 -> maxZ
    {
      const p1 = new THREE.Vector3()
        .addScaledVector(xDir, sinA * radius)
        .addScaledVector(yDir, cosA * radius)
        .addScaledVector(zDir, maxZ)
        .add(center);
      ring.push(p1);
    }

    rings.push(ring);
  }

  // --- 2. Build per-angular-segment surface vertices ---
  const faces = [];

  for (let r = 0; r < circleSegments - 1; r++) {
    const r1 = r + 1;

    // This segment is ONE vertical quad strip
    const face = [];

    const p00 = rings[r][0];   // minZ, angle r
    const p01 = rings[r][1];   // maxZ, angle r
    const p10 = rings[r1][0];  // minZ, angle r+1
    const p11 = rings[r1][1];  // maxZ, angle r+1

    // Store in consistent order (quad)
    face.push(p00.clone());
    face.push(p01.clone());
    face.push(p11.clone());
    face.push(p10.clone());

    faces.push(face);
  }

  return faces;

  }


  /* triangulate a 3D cylindrical face */
  static triangulateCylindricalFace(vertices, holes, transform,radius,circleSegments=16)
  {
    const xDir = GeometryUtils._vector1;
    const yDir = GeometryUtils._vector2;
    const zDir = GeometryUtils._vector3;
    const center = GeometryUtils._vector4;

    transform.extractBasis(xDir,yDir,zDir);

    // We only need directions, not scale
    xDir.normalize();   // cylinder axis (local Z)
    yDir.normalize();   // reference direction (angle = 0)
    zDir.normalize();   // perpendicular direction

    center.setFromMatrixPosition(transform);


    let minZ = Infinity, maxZ = -Infinity;
    const angleVec = [];

    const rel_tmp = new THREE.Vector3();

    for (const v of vertices) {
      rel_tmp.set(v.x,v.y,v.z);
      rel_tmp.subVectors(v, center);
      const dz = rel_tmp.dot(zDir);
      const dx = rel_tmp.dot(xDir);
      const dy = rel_tmp.dot(yDir);

      const angle = Math.atan2(dx, dy) * 180 / Math.PI;
      angleVec.push(angle);

      if (dz < minZ) minZ = dz;
      if (dz > maxZ) maxZ = dz;
    }

    // unwrap angles to avoid jumps >180º
    const unwrapped = [angleVec[0]];
    for (let i = 1; i < angleVec.length; i++) {
      let diff = angleVec[i] - angleVec[i - 1];
      if (diff > 180) diff -= 360;
      else if (diff < -180) diff += 360;
      unwrapped.push(unwrapped[i - 1] + diff);
    }

    let normalizeAngle = (angle) => ((angle % 360) + 360) % 360;
    

    let startDeg = normalizeAngle(Math.min(...unwrapped));
    let endDeg   = normalizeAngle(Math.max(...unwrapped));

    let span = endDeg - startDeg;
    if (span < 0) span += 360;

    endDeg = startDeg + span;

    return GeometryUtils.buildCylindricalSurface(transform, startDeg, endDeg, minZ, maxZ, radius, circleSegments);

  }

  static intersectLinePlane(v1, v2, plane)
  {
    let v21 = v2.clone().sub(v1);

    let t = -(plane.normal.dot(v1) + plane.constant) / plane.normal.dot(v21);

    return v21.multiplyScalar(t).add(v1);
  }

  /**
   * Returns an orthogonal vector
   *
   * @param {THREE.Vector3} vector - the input vector
   * @param {THREE.Vector3} orthoVector - the output vector
   * @returns {THREE.Vector3} ortho - an orthogonal vector of the given vector.
   */
  static orthogonalVector(vector, orthoVector)
  {
    if (!(orthoVector instanceof THREE.Vector3))
    {
      orthoVector = new THREE.Vector3();
    }

    if (Math.abs(vector.x) > 0.1)
    {
      orthoVector.set(vector.y, -vector.x, vector.z);
    }
    else if (Math.abs(vector.y) > 0.1)
    {
      orthoVector.set(-vector.y, vector.x, vector.z);
    }
    else // (~0, ~0, z)
    {
      orthoVector.set(-vector.z, vector.y, vector.x);
    }
    return orthoVector.cross(vector);
  }

  /**
   * Calculates the area of a BufferGeometry
   *
   * @param {BufferGeometry} geometry - the BufferGeometry
   * @param {Matrix4} matrix - the matrix to apply to geometry vertices
   * @returns {Number} area - the area of the BufferGeometry
   */
  static getBufferGeometryArea(geometry, matrix)
  {
    const triangle = new THREE.Triangle();
    const position = this._vector1;
    const vertices = geometry.attributes.position.array;

    function getVertex(index)
    {
      position.x = vertices[3 * index];
      position.y = vertices[3 * index + 1];
      position.z = vertices[3 * index + 2];
      if (matrix) position.applyMatrix4(matrix);
      return position;
    }

    let area = 0;

    this.getBufferGeometryFaces(geometry, (va, vb, vc) =>
    {
      triangle.a.copy(getVertex(va));
      triangle.b.copy(getVertex(vb));
      triangle.c.copy(getVertex(vc));
      area += triangle.getArea();
    });
    return area;
  }

  static traverseBufferGeometryVertices(geometry, callback)
  {
    const position = this._vector1;
    const positions = geometry.attributes.position.array;
    for (let i = 0; i < positions.length; i += 3)
    {
      position.x = positions[i];
      position.y = positions[i + 1];
      position.z = positions[i + 2];

      callback(position);
    }
  }

  static getBufferGeometryVertices(geometry)
  {
    const positions = geometry.attributes.position.array;
    const vertices = [];
    for (let i = 0; i < positions.length; i += 3)
    {
      let x = positions[i];
      let y = positions[i + 1];
      let z = positions[i + 2];
      vertices.push(new THREE.Vector3(x, y, z));
    }
    return vertices;
  }

  static getBufferGeometryFaces(geometry, addFace)
  {
    const positions = geometry.attributes.position.array;
    if (geometry.index) // indexed geometry
    {
      let indices = geometry.index.array;
      for (let i = 0; i < indices.length; i += 3)
      {
        let va = indices[i];
        let vb = indices[i + 1];
        let vc = indices[i + 2];

        addFace(va, vb, vc);
      }
    }
    else // non indexed geometry
    {
      var vertexCount = positions.length / 3;
      for (let i = 0; i < vertexCount; i += 3)
      {
        let va = i;
        let vb = i + 1;
        let vc = i + 2;

        addFace(va, vb, vc);
      }
    }
  }

  /**
   * Simplified version of BufferGeometryUtils.mergeBufferGeometries
   *
   * @param {Array<BufferGeometry>} geometries
   * @param {Boolean} useGroups
   * @return {BufferAttribute}
   */
  static mergeBufferGeometries(geometries, useGroups = false)
  {
    const isIndexed = geometries[0].index !== null;
    const attributesUsed = new Set(Object.keys(geometries[0].attributes));
    const attributes = {};
    const mergedGeometry = new THREE.BufferGeometry();

    let offset = 0;

    for (let i = 0; i < geometries.length; ++i)
    {
      const geometry = geometries[i];
      let attributesCount = 0;

      // ensure that all geometries are indexed, or none

      if (isIndexed !== (geometry.index !== null))
      {
        console.error('Not common attributes');
        return null;
      }

      // gather attributes, exit early if they're different
      for (const name in geometry.attributes)
      {
        if (!attributesUsed.has(name))
        {
          console.error('Not common attributes');
          return null;
        }

        if (attributes[name] === undefined)
        {
          attributes[name] = [];
        }

        attributes[name].push(geometry.attributes[name]);
        attributesCount++;
      }

      // ensure geometries have the same number of attributes

      if (attributesCount !== attributesUsed.size)
      {
        console.error('Not all geometries have the same number of attributes.');
        return null;
      }

      // gather .userData

      mergedGeometry.userData.mergedUserData =
        mergedGeometry.userData.mergedUserData || [];
      mergedGeometry.userData.mergedUserData.push(geometry.userData);

      if (useGroups)
      {
        let count;

        if (isIndexed)
        {
          count = geometry.index.count;
        }
        else if (geometry.attributes.position !== undefined)
        {
          count = geometry.attributes.position.count;
        }
        else
        {
          console.error('Geometry has not an index or a position attribute');
          return null;
        }
        mergedGeometry.addGroup(offset, count, i);
        offset += count;
      }
    }

    // merge indices

    if (isIndexed)
    {
      let indexOffset = 0;
      const mergedIndex = [];

      for (let i = 0; i < geometries.length; ++i)
      {
        const index = geometries[i].index;

        for (let j = 0; j < index.count; ++j)
        {
          mergedIndex.push(index.getX(j) + indexOffset);
        }
        indexOffset += geometries[i].attributes.position.count;
      }
      mergedGeometry.setIndex(mergedIndex);
    }

    // merge attributes

    for (const name in attributes)
    {
      const mergedAttribute = this.mergeBufferAttributes(attributes[name]);

      if (!mergedAttribute)
      {
        console.error('Failed while merging the ' + name + ' attribute.');
        return null;
      }
      mergedGeometry.setAttribute(name, mergedAttribute);
    }
    return mergedGeometry;
  }

  /**
   * @param {Array<BufferAttribute>} attributes
   * @return {BufferAttribute}
   */
  static mergeBufferAttributes(attributes)
  {
    let TypedArray;
    let itemSize;
    let normalized;
    let arrayLength = 0;

    for (let i = 0; i < attributes.length; ++i)
    {
      const attribute = attributes[ i ];

      if (attribute.isInterleavedBufferAttribute)
      {
        console.error('InterleavedBufferAttributes are not supported.');
        return null;
      }

      if (TypedArray === undefined)
      {
        TypedArray = attribute.array.constructor;
      }

      if (TypedArray !== attribute.array.constructor)
      {
        console.error('BufferAttribute.array is not consistent.');
        return null;
      }

      if (itemSize === undefined)
      {
        itemSize = attribute.itemSize;
      }

      if (itemSize !== attribute.itemSize)
      {
        console.error('BufferAttribute.itemSize is not consistent.');
        return null;
      }

      if (normalized === undefined)
      {
        normalized = attribute.normalized;
      }

      if (normalized !== attribute.normalized)
      {
        console.error('BufferAttribute.normalized is not consistent.');
        return null;
      }
      arrayLength += attribute.array.length;
    }

    const array = new TypedArray(arrayLength);
    let offset = 0;

    for (let i = 0; i < attributes.length; ++i)
    {
      array.set(attributes[i].array, offset);
      offset += attributes[i].array.length;
    }
    return new THREE.BufferAttribute(array, itemSize, normalized);
  }
}

export { GeometryUtils };
