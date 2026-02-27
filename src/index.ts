import { useEffect, useState } from 'react'
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
 * Load a Tensorflow Lite Model from the given `.tflite` asset.
 *
 * * If you are passing in a `.tflite` model from your app's bundle using `require(..)`, make sure to add `tflite` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.tflite` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.tflite` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param delegate The delegate to use for computations. Uses the standard CPU delegate per default. The `core-ml` or `metal` delegates are GPU-accelerated, but don't work on every model.
 * @returns The loaded Model.
 */
export function loadTensorflowModel(
  source: ModelSource,
  delegate: TensorflowModelDelegate = 'default'
): Promise<TensorflowModel> {
  let uri: string
  if (typeof source === 'number') {
    const asset = Image.resolveAssetSource(source)
    uri = asset.uri
  } else if (typeof source === 'object' && 'url' in source) {
    uri = source.url
  } else {
    throw new Error(
      'TFLite: Invalid source passed! Source should be either a React Native require(..) or a `{ url: string }` object!'
    )
  }
  return factory.loadModel(uri, delegate)
}

/**
 * Load a Tensorflow Lite Model from the given `.tflite` asset into a React State.
 *
 * * If you are passing in a `.tflite` model from your app's bundle using `require(..)`, make sure to add `tflite` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.tflite` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.tflite` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param delegate The delegate to use for computations. Uses the standard CPU delegate per default. The `core-ml` or `metal` delegates are GPU-accelerated, but don't work on every model.
 * @returns The state of the Model.
 */
export function useTensorflowModel(
  source: ModelSource,
  delegate: TensorflowModelDelegate = 'default'
): TensorflowPlugin {
  const [state, setState] = useState<TensorflowPlugin>({
    model: undefined,
    state: 'loading',
  })

  useEffect(() => {
    const load = async (): Promise<void> => {
      try {
        setState({ model: undefined, state: 'loading' })
        const m = await loadTensorflowModel(source, delegate)
        setState({ model: m, state: 'loaded' })
      } catch (e) {
        console.error(`Failed to load Tensorflow Model!`, e)
        setState({ model: undefined, state: 'error', error: e as Error })
      }
    }
    load()
  }, [delegate, source])

  return state
}

// Re-export types
export type { TensorflowModel, TensorflowModelDelegate, Tensor }
