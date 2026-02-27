import type { HybridObject } from 'react-native-nitro-modules'

/**
 * Tensor metadata describing an input or output tensor of a TFLite model.
 */
export interface Tensor {
  /**
   * The name of the Tensor.
   */
  name: string
  /**
   * The data-type all values of this Tensor are represented in.
   */
  dataType:
    | 'bool'
    | 'uint8'
    | 'int8'
    | 'int16'
    | 'int32'
    | 'int64'
    | 'float16'
    | 'float32'
    | 'float64'
    | 'invalid'
  /**
   * The shape of the data from this tensor.
   */
  shape: number[]
}

/**
 * The delegate used for TFLite inference acceleration.
 */
export type TensorflowModelDelegate =
  | 'default'
  | 'metal'
  | 'core-ml'
  | 'nnapi'
  | 'android-gpu'

type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array

/**
 * A loaded TFLite model backed by a Nitro HybridObject.
 */
export interface TensorflowModel extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  /**
   * The computation delegate used by this Model.
   */
  delegate: TensorflowModelDelegate
  /**
   * Run the Tensorflow Model with the given input buffer.
   * The input buffer has to match the input tensor's shape.
   */
  run(input: TypedArray[]): Promise<TypedArray[]>
  /**
   * Synchronously run the Tensorflow Model with the given input buffer.
   * The input buffer has to match the input tensor's shape.
   */
  runSync(input: TypedArray[]): TypedArray[]
  /**
   * All input tensors of this Tensorflow Model.
   */
  inputs: Tensor[]
  /**
   * All output tensors of this Tensorflow Model.
   */
  outputs: Tensor[]
}

/**
 * Factory HybridObject for loading TFLite models.
 */
export interface TfliteModelFactory extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  loadModel(path: string, delegate: string): Promise<TensorflowModel>
}
