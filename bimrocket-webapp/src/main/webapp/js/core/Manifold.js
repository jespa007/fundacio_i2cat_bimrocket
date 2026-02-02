import { SolidGeometry } from "./SolidGeometry.js";
import Module from "manifold";
import * as THREE from "three";


let api = undefined;
let wasm = undefined;

/*
function using(resource, fn) {
  try {
    return fn(resource);
  } finally {
    resource?.dispose?.();
  }
}*/

function ownedManifold(raw) {
  let alive = true;

  return Object.freeze({
    get ptr() { return raw.ptr; }, // always reflects current raw state

    dispose() {
      if (!alive) return;
      alive = false;
      api.destructAndFreeManifold(raw);
    }
  });
}

function ownedMeshgl(raw) {
  let alive = true;

  return Object.freeze({
    get ptr() { return raw.ptr; }, // always reflects current raw state

    shift_bytes: raw.shift_bytes,

    dispose() {
      if (!alive) {
        console.warn(`${label} already disposed`);
        return;
      }
      alive = false;
      api.destructAndFreeMeshgl(raw);
      console.log("meshgl deallocated");
    }
  });
}

async function initApi() {
  if (api) return; // already initialized

  const module_wasm = await Module();
  wasm = {
      module : module_wasm
      // Native bindings from manifold lib. They should NOT be called directly
      ,manifold_manifold_size: module_wasm._manifold_manifold_size
      ,manifold_destruct_manifold: module_wasm._manifold_destruct_manifold

      // meshgl (32 bit)
      ,meshgl32:{
         manifold_meshgl_size : module_wasm._manifold_meshgl_size
        ,manifold_meshgl: module_wasm._manifold_meshgl // set vertices, indexes
        ,manifold_get_meshgl: module_wasm._manifold_get_meshgl
        ,manifold_of_meshgl: module_wasm._manifold_of_meshgl
        ,manifold_meshgl_num_vert: module_wasm._manifold_meshgl_num_vert
        ,manifold_meshgl_num_tri: module_wasm._manifold_meshgl_num_tri
        ,manifold_meshgl_tri_verts: module_wasm._manifold_meshgl_tri_verts // triangle / indices
        ,manifold_meshgl_vert_properties: module_wasm._manifold_meshgl_vert_properties // triangles
        ,manifold_destruct_meshgl: module_wasm._manifold_destruct_meshgl
      }

      // meshgl64 (64 bit)
      ,meshgl64:{
         manifold_meshgl_size : module_wasm._manifold_meshgl64_size
        ,manifold_meshgl: module_wasm._manifold_meshgl64 // set vertices, indexes
        ,manifold_get_meshgl: module_wasm._manifold_get_meshgl64
        ,manifold_of_meshgl: module_wasm._manifold_of_meshgl64
        ,manifold_meshgl_num_vert: module_wasm._manifold_meshgl64_num_vert
        ,manifold_meshgl_num_tri: module_wasm._manifold_meshgl64_num_tri
        ,manifold_meshgl_tri_verts: module_wasm._manifold_meshgl64_tri_verts // triangle / indices
        ,manifold_meshgl_vert_properties: module_wasm._manifold_meshgl64_vert_properties // triangles
        ,manifold_destruct_meshgl: module_wasm._manifold_destruct_meshgl64 // destruct
      }

      // boolean ops.
      ,manifold_union: module_wasm._manifold_union
      ,manifold_difference: module_wasm._manifold_difference
      ,manifold_intersection: module_wasm._manifold_intersection
      ,manifold_volume: module_wasm._manifold_volume

      ,malloc: module_wasm._malloc
      ,free: module_wasm._free
  }

  api = {
    // manifold
    allocRawManifold: ()=>{
        let manifold =  wasm.malloc(wasm.manifold_manifold_size());

        if (manifold === 0) {
          throw new Error("allocRawManifold : cannot allocate memory");
        }

        return {ptr:manifold};
    }
    ,getManifoldVolume : (_manifold) => {
      return wasm.manifold_volume(_manifold.ptr);
    }
    ,destructAndFreeManifold : (_manifold)=>{
        wasm.manifold_destruct_manifold(_manifold.ptr);
        wasm.free(_manifold.ptr);
        _manifold.ptr = null;
    }
    ,boolean: (op, a, b) => {

      // op: "union" | "difference" | "intersection"

      if (a.ptr === null || b.ptr === null) {
         throw new Error(`boolean(${op}) : input manifold disposed`);
      }

      const dst = api.allocRawManifold();

      const fn = wasm[`manifold_${op}`];

      if (typeof fn !== "function") {
        throw new Error(`boolean: unknown op "${op}"`);
      }      

      const ret = fn(dst.ptr, a.ptr, b.ptr);

      if (ret !== dst.ptr) {
        throw new Error(`boolean(${op}) : unexpected address`);
      }

      return ownedManifold(dst);
    }
    ,getMeshglBindings(_shift_bytes){
      const size_bits = 8 << _shift_bytes;
      const meshgl_ops = wasm[`meshgl${size_bits}`];
      
      if (!meshgl_ops) throw new Error(`getMeshglBindings : meshgl${size_bits} bindings missing`);

      return meshgl_ops;
    }
    //-----------------------------------
    // meshgl
    ,allocMeshglObjectBuffer: (_shift_bytes)=>{
        const size_bits = 8 << _shift_bytes;
        const meshgl_ops = api.getMeshglBindings(_shift_bytes);

        let meshgl =  wasm.malloc(meshgl_ops.manifold_meshgl_size());

        if (meshgl === 0) {
          throw new Error(`allocMeshglObjectBuffer${size_bits} : cannot allocate memory`);
        }

        return { ptr: meshgl, shift_bytes: _shift_bytes };
    }
    ,getHeapArrays(_shift_bytes){
      const size_bits = 8 << _shift_bytes;
      return {
          HEAPF : size_bits === 64 ? wasm.module.HEAPF64 : wasm.module.HEAPF32
          ,HEAPU : size_bits === 64 ? wasm.module.HEAPU64 : wasm.module.HEAPU32
      }
    }
    ,createMeshglFromRawArraysInternal: (_params, _shift_bytes)=>{

      const meshgl_ops = api.getMeshglBindings(_shift_bytes);

      const size_bits = 8 << _shift_bytes;
      const size_bytes = 1 << _shift_bytes;

      if (_params.indices.length === 0) {
        throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : No triangles`);
      }

      if (_params.positions.length % 3 !== 0) {
        throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : positions length not multiple of 3`);
      }

      if (_params.indices.length % 3 !== 0) {
        throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : indices length not multiple of 3`);
      }

      switch(size_bits){
      case 64:
        if (!(_params.indices instanceof BigUint64Array)) {
          throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : meshgl requires BigUint64Array indices`);
        }
        if (!(_params.positions instanceof Float64Array)) {
          throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : meshgl requires Float64Array positions`);
        }          
        break;

      case 32:
        if (!(_params.indices instanceof Uint32Array)) {
          throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : meshgl requires Uint32Array indices`);
        }
        if (!(_params.positions instanceof Float32Array)) {
          throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : meshgl requires Float32Array positions`);
        }          
        break;
      }

      let meshgl = api.allocMeshglObjectBuffer(_shift_bytes);

      const vertexCount = _params.positions.length / 3;
      const triCount = _params.indices.length / 3;

      // Allocate buffer for upload data in WASM
      const vBytes = _params.positions.length * size_bytes; // float[size_bits]
      const iBytes = _params.indices.length * size_bytes;   // uint[size_bits]

      const vPtr = wasm.malloc(vBytes);
      const iPtr = wasm.malloc(iBytes);

      try {

        const {HEAPF,HEAPU} = api.getHeapArrays(_shift_bytes);

        // Copy data into WASM heap
        HEAPF.set(_params.positions, vPtr >> _shift_bytes);
        HEAPU.set(_params.indices, iPtr >> _shift_bytes);

        // construct mesh with indices/vertexes of MeshGL64
        const ret = meshgl_ops.manifold_meshgl(
            meshgl.ptr,
            vPtr,
            vertexCount,
            3, // xyz count
            iPtr,
            triCount
        );

        if (ret !== meshgl.ptr) {
          throw new Error(`createMeshglFromRawArraysInternal(${size_bits}) : meshgl constructed at unexpected address`);
        }            
      }catch(ex){
        console.error(ex);
      } finally {
        // deallocate pointers
        wasm.free(vPtr);
        wasm.free(iPtr);
      }

      return ownedMeshgl(meshgl);
    }
    ,createMeshgl64FromRawArrays(_params){
      return api.createMeshglFromRawArraysInternal(_params,3);
    }
    ,createMeshgl32FromRawArrays(_params){
      return api.createMeshglFromRawArraysInternal(_params,2);
    }
    ,manifoldToMeshgl : (_manifold, _shift_bytes = 3) => {
      const size_bits = 8 << _shift_bytes;

      if (!_manifold || _manifold.ptr === null) {
        throw new Error(`manifoldToMeshgl(${size_bits}): manifold is null or disposed`);
      }

      const meshgl_ops = api.getMeshglBindings(_shift_bytes);
      const rawMesh = api.allocMeshglObjectBuffer(_shift_bytes);

      const ret = meshgl_ops.manifold_get_meshgl(
        rawMesh.ptr,
        _manifold.ptr
      );

      if (ret !== rawMesh.ptr) {
        wasm.free(rawMesh.ptr);
        throw new Error(`manifoldToMeshgl(${size_bits}) : unexpected MeshGL address`);
      }

      return ownedMeshgl(rawMesh);
    }
    ,manifoldToMeshgl32 : (_manifold) => {
      return api.manifoldToMeshgl(_manifold, 2); // 32bits
    }
    ,manifoldToMeshgl64 : (_manifold) => {
      return api.manifoldToMeshgl(_manifold, 3); // 64bits
    }
    ,meshgltoManifold: (_meshgl) => {

      if (_meshgl.ptr === null) {
         throw new Error("meshgltoManifold : MeshGL already disposed");
      }

      const meshgl_ops = api.getMeshglBindings(_meshgl.shift_bytes);

      let manifold = api.allocRawManifold();
      const ret = meshgl_ops.manifold_of_meshgl(manifold.ptr, _meshgl.ptr);

      if (ret !== manifold.ptr) {
          throw new Error(`meshgltoManifold : unexpected address`);
      }   

      return ownedManifold(manifold);
    }
    ,meshglToRawArrays: (_meshgl)=>{

       const size_bits = 8 << _meshgl.shift_bytes;
       const size_bytes = 1 << _meshgl.shift_bytes;

       if (_meshgl.ptr === null) {
         throw new Error(`meshToRawArrays(${size_bits}) : MeshGL already disposed`);
      }

        const meshgl_ops = api.getMeshglBindings(_meshgl.shift_bytes);



        let positions = undefined;
        let indices = undefined;

        // Query sizes
        const vertexCount = meshgl_ops.manifold_meshgl_num_vert(_meshgl.ptr);
        const triCount = meshgl_ops.manifold_meshgl_num_tri(_meshgl.ptr);

        if ((vertexCount === 0 || triCount === 0 ) === true)
        {
          throw new Error(`meshToRawArrays(${size_bits}) : mesh has no data`);
        }

        const numProps = 3; // xyz
        const indexArity = 3; // i0,i1,i2

        // Allocate buffer to get indices / vertices data from WASM
        const vBytes = vertexCount * size_bytes * numProps; // float-size_bits
        const iBytes = triCount * size_bytes* indexArity;    // uint-size_bits

        const vPtr = wasm.malloc(vBytes);
        const iPtr = wasm.malloc(iBytes);

        try{
            
            // Copy result to tmp positions / indices
            meshgl_ops.manifold_meshgl_vert_properties(vPtr,_meshgl.ptr);
            meshgl_ops.manifold_meshgl_tri_verts(iPtr,_meshgl.ptr);

            const FloatXXArray = size_bits === 64 ? Float64Array : Float32Array;
            const UintXXArray = size_bits === 64 ? BigUint64Array : Uint32Array;
            const {HEAPF,HEAPU} = api.getHeapArrays(_meshgl.shift_bytes);

            // COPY data out of WASM
            positions = new FloatXXArray(
              HEAPF.buffer,
              vPtr,
              vertexCount*numProps
            ).slice();

            const tri = new UintXXArray(
              HEAPU.buffer
              , iPtr
              , triCount* indexArity
            );

            if(size_bits === 64){
              indices = new Uint32Array(triCount* indexArity);
              for (let i = 0; i < indices.length; i++) {
                const x = tri[i];
                if (x > 0xffffffffn) {
                  throw new Error(`meshToRawArrays(${size_bits}) : index > 32-bit`);
                }
                indices[i] = Number(x);
              }
            }else{
              indices = tri.slice();
            }
        }catch(ex){
          console.error(ex);
        }finally{
          // Free buffers and mesh
          wasm.free(vPtr);
          wasm.free(iPtr);
        }        
        return [indices, positions];
    }
    ,destructAndFreeMeshgl : (_meshgl)=>{

      const size_bits = 8 << _meshgl.shift_bytes;
       if (_meshgl.ptr === null) {
         throw new Error(`destructAndFreeMeshgl(${size_bits}) : MeshGL already disposed`);
      }

      const meshgl_ops = api.getMeshglBindings(_meshgl.shift_bytes);

      meshgl_ops.manifold_destruct_meshgl(_meshgl.ptr);
      wasm.free(_meshgl.ptr);

      _meshgl.ptr = null;

    }

  };

  return api;
}

function extractMeshGLParamsFromSolidGeometry(
	solidGeometry
	, worldMatrix
	, eps = 1e-6 // 1e-6 => meters
) {

  const srcPos = solidGeometry.attributes.position.array;

  const map = new Map();            // quantized position → new index
  const newPos = [];                // Float64 positions
  const newIdx = [];                // triangle indices

  const v = new THREE.Vector3();
  let nextIndex = 0;

  const quantKey = (x, y, z) => {
    return (
      Math.round(x / eps) + "," +
      Math.round(y / eps) + "," +
      Math.round(z / eps)
    );
  };

  // iterate triangles (SolidGeometry is non-indexed)
  for (let i = 0; i < srcPos.length; i += 9) {
    const triIds = [];

    for (let k = 0; k < 3; k++) {
      const base = i + k * 3;
      v.fromArray(srcPos, base);
      v.applyMatrix4(worldMatrix);

      const key = quantKey(v.x, v.y, v.z);

      let id = map.get(key);
      if (id === undefined) {
        id = nextIndex++;
        map.set(key, id);
        newPos.push(v.x, v.y, v.z);
      }
      triIds.push(id);
    }

    // skip degenerate triangles created by welding
    if (triIds[0] === triIds[1] || triIds[0] === triIds[2] || triIds[1] === triIds[2]) {
      continue;
    }

    newIdx.push(triIds[0], triIds[1], triIds[2]);
  }

  const indices64 = new BigUint64Array(newIdx.length );
    for (let i = 0; i < newIdx.length ; i++) {
      indices64[i] = BigInt(newIdx[i]);
    }

  return {
    positions: new Float64Array(newPos),
    indices: indices64
  };
}

  /*if (!solidGeometry) {
    throw new Error("fromSolidGeometry: solidGeometry is null");
  }

  const posAttr = solidGeometry.attributes?.position;
  if (!posAttr) {
    throw new Error("fromSolidGeometry: geometry has no position attribute");
  }

  const srcPos = posAttr.array;
  const vertexCount = srcPos.length / 3;

  if (vertexCount % 3 !== 0) {
    throw new Error("fromSolidGeometry: vertex count not divisible by 3");
  }

  // ---- Step 1: build transformed positions (Float64)
  const positions = new Float64Array(srcPos.length);
  const v = new THREE.Vector3();

  for (let i = 0; i < vertexCount; i++) {
    const i3 = i * 3;

    v.set(
      srcPos[i3],
      srcPos[i3 + 1],
      srcPos[i3 + 2]
    );

    if (worldMatrix) {
      v.applyMatrix4(worldMatrix);
    }

    positions[i3]     = v.x;
    positions[i3 + 1] = v.y;
    positions[i3 + 2] = v.z;
  }

  // ---- Step 2: build indices (triangle soup → sequential)
  const triCount = vertexCount / 3;
  const indices = new BigUint64Array(vertexCount);

  for (let i = 0; i < vertexCount; i++) {
    indices[i] = BigInt(i);
  }*/

function fromSolidGeometry(solidGeometry, worldMatrix = null){

  const params = extractMeshGLParamsFromSolidGeometry(solidGeometry, worldMatrix);

  // ---- Step 1: create MeshGL64
  const meshgl = api.createMeshgl64FromRawArrays(params);
  let manifold;

  // ---- Step 2: convert to Manifold (consume meshgl safely)
  try{
      manifold = api.meshgltoManifold(meshgl);
  }finally{
      meshgl.dispose();
  }
  
  if (!manifold) {
    // empty or invalid manifold
    return null;
  }

  const vol = api.getManifoldVolume(manifold);
  if (Math.abs(vol) < 1e-12){
      manifold.dispose();
      return null;
  }


  return manifold; // ownedManifold
}

function toSolidGeometry(manifold)
{
  
  const  result = new SolidGeometry(); // empty

  if (!manifold || manifold.ptr === null) {
    return result;
  }

  let meshgl = api.manifoldToMeshgl64(manifold);

  if(!meshgl){
    return result;
  }

  let positions;
  let indices;
  
  try{
    [indices, positions] = api.meshglToRawArrays(meshgl);
  }finally{
    meshgl.dispose();
  }

  

    
  // code to implement ...

  if(positions !== undefined){
    // Build Three.js BufferGeometry
    //const geometry = new THREE.BufferGeometry();
    //const numProps = 3; // xyz

    // Convert to float32 for Three.js
   /* geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        new Float32Array(positions),
        numProps
      )
    );

    geometry.setIndex(
      new THREE.BufferAttribute(indices, 1)
    );*/

   // geometry.computeVertexNormals();

    // Convert to SolidGeometry
    const pos = positions;   // FloatXXArray
    const idx = indices;                 // UintXXArray

    for (let i = 0; i < idx.length; i += 3) {
      const ia = idx[i] * 3;
      const ib = idx[i + 1] * 3;
      const ic = idx[i + 2] * 3;

      const a = new THREE.Vector3(pos[ia], pos[ia + 1], pos[ia + 2]);
      const b = new THREE.Vector3(pos[ib], pos[ib + 1], pos[ib + 2]);
      const c = new THREE.Vector3(pos[ic], pos[ic + 1], pos[ic + 2]);

      result.addFace(a, b, c); // or whatever SolidGeometry expects
    } 

    result.updateBuffers();
  }

  return result;

}

// Booleans
// op: "union" | "difference" | "intersection"
function union(a, b) {
	return api.boolean("union",a, b);
}

function subtract(a, b) {
	return api.boolean("difference",a, b);
}

function intersect(a, b) {
	return api.boolean("intersection",a, b);
}

 // Free manifold resource



export {initApi,fromSolidGeometry,toSolidGeometry,union,subtract,intersect}