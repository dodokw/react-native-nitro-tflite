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
 *
 * - `default`      — CPU inference (all platforms)
 * - `core-ml`      — Apple CoreML (iOS, requires `$EnableCoreMLDelegate` in Podfile)
 * - `metal`        — Apple Metal GPU (iOS, requires `$EnableMetalDelegate` in Podfile)
 * - `nnapi`        — Android Neural Networks API (Android only)
 * - `android-gpu`  — Android GPU delegate (Android only)
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
   * Run the Tensorflow Model with the given input buffer asynchronously.
   * The input buffer has to match the input tensor's shape.
   */
  run(input: TypedArray[]): Promise<TypedArray[]>
  /**
   * Synchronously run the Tensorflow Model with the given input buffer.
   * The input buffer has to match the input tensor's shape.
   * Prefer this in frame processors or performance-critical paths.
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
  /**
   * Dynamically resize an input tensor.
   *
   * Call this before `run`/`runSync` when the model supports dynamic shapes.
   * After calling `reshapeInput`, the input and output tensor sizes are
   * re-allocated automatically.
   *
   * @param inputIndex  Zero-based index of the input tensor to resize.
   * @param shape       The new shape, e.g. `[1, 320, 320, 3]`.
   *
   * @example
   * model.reshapeInput(0, [1, 640, 640, 3])
   * const output = model.runSync([inputFloat32])
   */
  reshapeInput(inputIndex: number, shape: number[]): void
}

/**
 * Factory HybridObject for loading TFLite models.
 * @internal - Use `loadTensorflowModel` from the public API instead.
 */
export interface TfliteModelFactory extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  /**
   * Load a TFLite model from the given URI.
   *
   * @param path       The model URI (file:// or http(s)://).
   * @param delegate   The inference delegate to use.
   * @param onProgress Optional callback for download progress in [0, 1].
   *                   Receives -1 when content-length is unknown.
   */
  loadModel(
    path: string,
    delegate: TensorflowModelDelegate,
    onProgress?: (progress: number) => void
  ): Promise<TensorflowModel>
  /**
   * Clear the in-memory model cache.
   * Cached entries are weak references and are automatically released when
   * the model is GC'd, but this method lets you eagerly free memory.
   */
  clearCache(): void
}
