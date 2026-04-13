import { useCallback, useEffect, useRef, useState } from 'react'
import { Image } from 'react-native'
import { NitroModules } from 'react-native-nitro-modules'
import type {
  TensorflowModel,
  TensorflowModelDelegate,
  TfliteModelFactory,
  Tensor,
} from './specs/TfliteTypes'

// Create the factory HybridObject from the Nitro registry
const factory = NitroModules.createHybridObject<TfliteModelFactory>('TfliteModelFactory')

// In React Native, `require(..)` returns a number.
type Require = number
type ModelSource = Require | { url: string }

/**
 * Callback fired during model loading to report download progress.
 * - `progress` is in the range [0, 1].
 * - `progress === -1` means the content-length is unknown (indeterminate).
 */
export type ModelLoadProgress = (progress: number) => void

export type TensorflowPlugin =
  | {
      model: TensorflowModel
      state: 'loaded'
    }
  | {
      model: undefined
      state: 'loading'
    }
  | {
      model: undefined
      error: Error
      state: 'error'
    }

/**
 * Resolve a {@link ModelSource} to a URI string understood by the native layer.
 */
function resolveSourceUri(source: ModelSource): string {
  if (typeof source === 'number') {
    return Image.resolveAssetSource(source).uri
  }
  if (typeof source === 'object' && 'url' in source) {
    return source.url
  }
  throw new Error(
    'TFLite: Invalid source! Pass either a require(..) number or a { url: string } object.'
  )
}

/**
 * Load a TensorFlow Lite model from the given `.tflite` asset.
 *
 * Results are cached in native memory — calling this function twice with the
 * same `source` and `delegate` returns the **same underlying model** without
 * re-loading or re-allocating tensors.
 *
 * * If you are passing in a `.tflite` model from your app's bundle using `require(..)`,
 *   make sure to add `tflite` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, the URL can be a web URL (`http://`/`https://`)
 *   or a local file (`file://`).
 *
 * @param source      The model source — either `require('./model.tflite')` or `{ url: '...' }`.
 * @param delegate    The inference delegate. Defaults to `'default'` (CPU).
 * @param onProgress  Optional callback reporting download progress in [0, 1].
 *                    Receives `-1` when `Content-Length` is unknown.
 * @returns           A promise that resolves to the loaded {@link TensorflowModel}.
 */
export function loadTensorflowModel(
  source: ModelSource,
  delegate: TensorflowModelDelegate = 'default',
  onProgress?: ModelLoadProgress
): Promise<TensorflowModel> {
  const uri = resolveSourceUri(source)
  return factory.loadModel(uri, delegate, onProgress)
}

/**
 * Eagerly clear the native model cache.
 *
 * Cached entries are automatically released when the JS model object is GC'd,
 * so calling this is optional. Use it to free memory proactively (e.g. when
 * navigating away from a screen that loaded multiple large models).
 */
export function clearTensorflowModelCache(): void {
  factory.clearCache()
}

/**
 * Load a TensorFlow Lite model into React state.
 *
 * Results are cached in native memory — changing `source` or `delegate` will
 * re-load the model, but calling with identical values returns the cached one.
 *
 * * If you are passing in a `.tflite` model using `require(..)`, add `tflite`
 *   as an asset extension in `metro.config.js`.
 * * If you are passing in a `{ url: ... }`, the URL can be `http://`, `https://`,
 *   or `file://`.
 *
 * @param source    The model source.
 * @param delegate  The inference delegate. Defaults to `'default'` (CPU).
 * @returns         The current load state of the model.
 */
export function useTensorflowModel(
  source: ModelSource,
  delegate: TensorflowModelDelegate = 'default'
): TensorflowPlugin {
  const [state, setState] = useState<TensorflowPlugin>({
    model: undefined,
    state: 'loading',
  })

  // Stable dependency key — avoids re-loading when callers inline `{ url }` objects.
  const sourceKey = typeof source === 'number' ? source : source.url

  // Keep a ref to the latest setState so the async callback never closes over a stale value.
  const setStateRef = useRef(setState)
  setStateRef.current = setState

  const load = useCallback(async (): Promise<void> => {
    setStateRef.current({ model: undefined, state: 'loading' })
    try {
      const m = await loadTensorflowModel(source, delegate)
      setStateRef.current({ model: m, state: 'loaded' })
    } catch (e) {
      console.error('Failed to load Tensorflow Model!', e)
      setStateRef.current({ model: undefined, state: 'error', error: e as Error })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, delegate])

  useEffect(() => {
    load()
  }, [load])

  return state
}

// Re-export types
export type { TensorflowModel, TensorflowModelDelegate, Tensor, ModelSource }
