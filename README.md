# react-native-nitro-tflite

High-performance TensorFlow Lite library for React Native, powered by **Nitro Modules**.

> **Note:** This is an unofficial Nitro Module migration of [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite), fully optimized for the New Architecture, Bridgeless mode, and Vision Camera frame processors.

This library provides the same API as the original, with the following improvements:

- ✅ **HybridObject-based** — No more `install()` or global JSI functions
- ✅ **Works from any frame processor/worklet** — Can be called from any thread/runtime
- ✅ **Bridge/Bridgeless agnostic** — Works on both architectures automatically
- ✅ **Same API** — Drop-in replacement, same `loadTensorflowModel()` and `useTensorflowModel()`

## Installation

```sh
npm install react-native-nitro-tflite react-native-nitro-modules
```

### iOS

```sh
cd ios && pod install
```

### Android

No additional steps needed. The TFLite libraries are automatically downloaded via Gradle.

## Usage

```tsx
import { useTensorflowModel } from 'react-native-nitro-tflite'

function App() {
  const model = useTensorflowModel(require('./model.tflite'))

  if (model.state === 'loaded') {
    console.log('Inputs:', model.model.inputs)
    console.log('Outputs:', model.model.outputs)

    // Run inference
    const output = model.model.runSync([inputData])
  }

  return <View />
}
```

### Loading from URL

```ts
import { loadTensorflowModel } from 'react-native-nitro-tflite'

const model = await loadTensorflowModel(
  { url: 'https://example.com/model.tflite' },
  'default'
)
```

### Delegates

| Delegate      | Platform | Description                   |
| ------------- | -------- | ----------------------------- |
| `default`     | Both     | CPU inference                 |
| `core-ml`     | iOS      | Apple CoreML acceleration     |
| `metal`       | iOS      | Metal GPU (not yet supported) |
| `nnapi`       | Android  | Android Neural Networks API   |
| `android-gpu` | Android  | Android GPU delegate          |

## Architecture

This library uses [Nitro Modules](https://nitro.margelo.com) with manual C++ `HybridObject` implementation:

- **`HybridTfliteModel`** — Wraps `TfLiteInterpreter` with `run`/`runSync`/`inputs`/`outputs`/`delegate`
- **`HybridTfliteModelFactory`** — Factory for loading models with platform-specific URL fetching

```
JS:  loadTensorflowModel()
       ↓
     NitroModules.createHybridObject("TfliteModelFactory")
       ↓
     factory.loadModel(url, delegate)
       ↓
C++: HybridTfliteModelFactory::loadModelRaw()
       ↓
     TfLiteModelCreate() + TfLiteInterpreterCreate()
       ↓
     new HybridTfliteModel(interpreter)  ← returned to JS as HybridObject
```

## Metro Configuration

Add `tflite` as an asset extension in your `metro.config.js`:

```js
const { getDefaultConfig } = require('@react-native/metro-config')

const config = getDefaultConfig(__dirname)
config.resolver.assetExts.push('tflite')

module.exports = config
```

## License

MIT
