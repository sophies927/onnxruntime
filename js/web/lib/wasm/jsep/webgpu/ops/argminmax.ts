// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length === 0 || inputs.length > 2) {
    throw new Error('Reduce op requires 1 or 2 inputs.');
  }
  if (inputs.length === 2 && inputs[1].dims.length !== 1) {
    throw new Error('Invalid axes input dims.');
  }
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Invalid input type.');
  }
};

export interface ReduceAttributes extends AttributeWithCacheKey {
  keepDims: boolean;
  noopWithEmptyAxes: boolean;
  axes: number[];
}

type ArgMinMaxOp = (inputs: readonly TensorView[], axes: number[]) => string[];

const noOp: ArgMinMaxOp = (): string[] => ['', '', 'value = _A[inputIdx];', ''];

const createReduceProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: ReduceAttributes,
     argMinMaxOp: ArgMinMaxOp): ProgramInfo => {
      const outputShape: number[] = [];
      const inputShape = inputs[0].dims;

      const idxCopy: string[] = [];  // copy output indexes to input indexes

      const axes = ShapeUtil.normalizeAxes(attributes.axes, inputs[0].dims.length);
      const outputDimsLength = inputs[0].dims.length - (attributes.keepDims ? 0 : axes.length);
      const ops = argMinMaxOp(inputs, axes);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const initInputIdx = (ops[1] === '') ? '' : `let inputIdx = ${inputIndicesHelper.i2oExpression('inputIndices')};`;
      let reduceOps = `
          let inputIdx = ${inputIndicesHelper.i2oExpression('inputIndices')};
          ${ops[2]};`;
      const reduceOnAllAxes = !attributes.noopWithEmptyAxes && attributes.axes.length === 0;
      for (let k = 0; k < inputs[0].dims.length; k++) {
        // if this axis is reduced
        if (reduceOnAllAxes || axes.indexOf(k) >= 0) {
          if (attributes.keepDims) {
            outputShape.push(1);
          }  // else { remove the axis from outputShape; }

          // loop over the d-th axis
          reduceOps = `for(var j${k}: u32 = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
                            let lastIndex = j${k};
                            inputIndices[${k}] = lastIndex;
                            ${reduceOps}
                          }`;
        } else {
          if (outputDimsLength > 1) {
            idxCopy.push(`inputIndices[${k}] = outputIndices[${outputShape.length}];`);
          } else {
            idxCopy.push(`inputIndices[${k}] = outputIndices;`);
          }
          outputShape.push(inputs[0].dims[k]);
        }
      }

      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const outputSize = ShapeUtil.size(outputShape);
      const dataType = 'f32';

      const getShaderSource = (shaderHelper: ShaderHelper) => `
          @group(0) @binding(0) var<storage, read> _A : array<${dataType}>;
          @group(0) @binding(1) var<storage, read_write> output : array<i32>;

          ${outputIndicesHelper.o2iImpl}
          ${inputIndicesHelper.i2oImpl}

          ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
          ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
          ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}

          ${idxCopy.join('\n')}
          ${ops[0]}       // init ops
          ${initInputIdx}
          ${ops[1]}
          ${reduceOps}
          ${ops[3]} // final
          output[global_idx*2] = bestIndex;
        }`;

      return {
        ...metadata,
        getShaderSource,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

const createReduceAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: ReduceAttributes): ReduceAttributes => {
      const axes: number[] = [];
      if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => axes.push(Number(v)));
      }
      return createAttributeWithCacheKey(
          {axes, keepDims: attributes.keepDims, noopWithEmptyAxes: attributes.noopWithEmptyAxes});
    };

const createReduceProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: ReduceAttributes, reduceOp: ArgMinMaxOp):
        ProgramInfoLoader => {
          const updatedAttributes: ReduceAttributes =
              inputs.length === 1 ? attributes : createReduceAttributesFromInputs(inputs, attributes);
          const metadata:
              ProgramMetadata = {name, inputTypes: [GpuDataType.default], cacheHint: updatedAttributes.cacheKey};
          return {
            ...metadata,
            get: () => createReduceProgramInfo(
                metadata, [inputs[0]], updatedAttributes,
                updatedAttributes.noopWithEmptyAxes && updatedAttributes.axes.length === 0 ? noOp : reduceOp)
          };
        };


export const argMin = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const argMinMaxOp: ArgMinMaxOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }

    return [
      `${idxZero.join('\n')}`,
      'var value = _A[inputIdx];\nvar bestIndex : i32 = 0;',
      'if (_A[inputIdx] < value) {value = _A[inputIdx]; bestIndex = i32(lastIndex);} ',
    ];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ArgMin', attributes, argMinMaxOp), {inputs: [0]});
};

export const argMax = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const argMinMaxOp: ArgMinMaxOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }

    return [
      `${idxZero.join('\n')}`,
      'var value = _A[inputIdx];\nvar bestIndex : i32 = 0;',
      'if (_A[inputIdx] > value) {value = _A[inputIdx]; bestIndex = i32(lastIndex);} ',
      ''
    ];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'argMax', attributes, argMinMaxOp), {inputs: [0]});
};
