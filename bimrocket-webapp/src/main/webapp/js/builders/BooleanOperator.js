/*
 * BooleanOperator.js
 *
 * @author realor
 */

import { ObjectBuilder } from "./ObjectBuilder.js";
import { SolidBuilder } from "./SolidBuilder.js";
import { SolidGeometry } from "../core/SolidGeometry.js";
import { Solid } from "../core/Solid.js";
import { BSP } from "../core/BSP.js";
import * as Manifold from "../core/Manifold.js";
import * as THREE from "three";

class BooleanOperator extends SolidBuilder
{
  static UNION = "union";
  static INTERSECT = "intersect";
  static SUBTRACT = "subtract";

  static BSP = "bsp";
  static MANIFOLD  = "manifold";

  constructor(operation, backend)
  {
    super();
    this.operation = operation || BooleanOperator.SUBTRACT;
    this.backend = backend || BooleanOperator.BSP; // default
  }

  performBuildBSP(solid, solids) {
    const matrix = new THREE.Matrix4();

    const createBSP = child =>
    {
      matrix.copy(child.matrix);

      let parent = child.parent;
      while (parent && parent !== solid)
      {
        matrix.premultiply(parent.matrix);
        parent = parent.parent;
      }
      const bsp = new BSP();
      bsp.fromSolidGeometry(child.geometry, matrix);
      return bsp;
    };

    let resultBSP = createBSP(solids[0]);

    for (let i = 1; i < solids.length; i++) {
      let solid = solids[i];
      if (solid.isValid())
      {
        let otherBSP = createBSP(solid);
        switch (this.operation)
        {
          case BooleanOperator.UNION:
            resultBSP = resultBSP.union(otherBSP); break;
          case BooleanOperator.INTERSECT:
            resultBSP = resultBSP.intersect(otherBSP); break;
          case BooleanOperator.SUBTRACT:
            resultBSP = resultBSP.subtract(otherBSP); break;
        }
      }
    }

    return resultBSP.toSolidGeometry();
  }

  performBuildManifold(solid, solids)
  {
    let geometry; // geoemtry output

    // Prepare
    if (solids.length < 2) {
      return;
    }

    const matrix = new THREE.Matrix4();

    const buildWorldMatrix = child =>
    {
      matrix.copy(child.matrix);
      let parent = child.parent;

      while (parent && parent !== solid)
      {
        matrix.premultiply(parent.matrix);
        parent = parent.parent;
      }

      return matrix.clone();
    };

    // Convert solids → Manifold handles ---
    const manifolds = [];

    try
    {
      for (let s of solids)
      {
        if (!s.isValid()) continue;

        const worldMatrix = buildWorldMatrix(s);

        const h = Manifold.fromSolidGeometry(
          s.geometry,
          worldMatrix
        );

        if(!h){
          manifolds.push(h);
        }
      }

      if (manifolds.length > 0) {

        // Apply boolean operation
        let result = manifolds[0];

        for (let i = 1; i < manifolds.length; i++)
        {
          const other = manifolds[i];

          switch (this.operation)
          {
            case BooleanOperator.UNION:
              result = Manifold.union(result, other);
              break;

            case BooleanOperator.INTERSECT:
              result = Manifold.intersect(result, other);
              break;

            case BooleanOperator.SUBTRACT:
              result = Manifold.subtract(result, other);
              break;
          }
        }

        // Convert result → SolidGeometry
        geometry = Manifold.toSolidGeometry(result);
      }
    }catch(ex){
      console.error(ex);
    }finally{
      // free manifold manifolds ---
      for (let manifold of manifolds)
      {
        if(manifold){
          manifold.dispose();
        }
      }
    }

    return geometry;
  }

  performBuild(solid)
  {
    const solids = [];
    this.findSolids(solid, solids);

    if (solids.length === 0) return true;

    let geometry;

    switch (this.backend){
      case BooleanOperator.MANIFOLD:
        geometry = this.performBuildManifold(solid, solids);
        break;
      case BooleanOperator.BSP:
      default:
        geometry = this.performBuildBSP(solid, solids);
        break;
    }

    if(!geometry){
      return false;
    }
    
    geometry.smoothAngle = this.calculateSmoothAngle(solids);
    solid.updateGeometry(geometry, true);

    return true;
  }

  copy(source)
  {
    this.operation = source.operation;

    return this;
  }

  calculateSmoothAngle(solids)
  {
    let smoothAngle = 0;
    for (let solid of solids)
    {
      if (solid.geometry.smoothAngle > smoothAngle)
      {
        smoothAngle = solid.geometry.smoothAngle;
      }
    }
    return smoothAngle;
  }

  findSolids(object, solids)
  {
    const children = object.children;
    const start = object instanceof Solid ? 2 : 0;
    for (let i = start; i < children.length; i++)
    {
      let child = children[i];
      if (child instanceof Solid)
      {
        child.visible = false;
        child.edgesVisible = false;
        child.facesVisible = false;
        solids.push(child);
      }
      else
      {
        this.findSolids(child, solids);
      }
    }
  }
}

ObjectBuilder.addClass(BooleanOperator);

export { BooleanOperator };

