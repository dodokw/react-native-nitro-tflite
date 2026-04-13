# react-native-nitro-tflite

High-performance TensorFlow Lite library for React Native, powered by **Nitro Modules**.

> **Note:** This is an unofficial Nitro Module migration of [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite), fully optimized for the New Architecture, Bridgeless mode, and Vision Camera frame processors.

## Features

- ✅ **HybridObject-based** — No `install()` or global JSI functions
- ✅ **Works from any frame processor / worklet** — Callable from any thread or runtime
- ✅ **Bridge / Bridgeless agnostic** — Works on both architectures automatically
- ✅ **Same API** — Drop-in replacement for `react-native-fast-tflite`
- ✅ **Model caching** — Identical calls return the same model with zero re-allocation
- ✅ **Progress callbacks** — Track large model downloads in real time
- ✅ **Dynamic input shapes** — Resize input tensors at runtime
- ✅ **Metal GPU delegate** (iOS) — Hardware-accelerated inference via Apple Metal

---

## Installation

```sh
npm install react-native-nitro-tflite react-native-nitro-modules
```

### iOS

```sh
cd ios && pod install
```

### Android

No additional steps needed — TFLite libraries are downloaded automatically via Gradle.

---

## Usage

### Basic — React hook

```tsx
import { useTensorflowModel } from 'react-native-nitro-tflite'

function App() {
  const model = useTensorflowModel(require('./model.tflite'))

  if (model.state === 'loaded') {
    const output = model.model.runSync([inputFloat32Array])
    console.log('Output:', output)
  }

  if (model.state === 'error') {
    console.error('Failed to load:', model.error)
  }

  return <View />
}
```

### Load imperatively

```ts
import { loadTensorflowModel } from 'react-native-nitro-tflite'

const model = await loadTensorflowModel(require('./model.tflite'))
const output = model.runSync([inputData])
```

### Load from URL

```ts
const model = await loadTensorflowModel({ url: 'https://example.com/model.tflite' })
```

---

## Progress Callback

Track download progress for large models (useful for remote URLs):

```ts
import { loadTensorflowModel } from 'react-native-nitro-tflite'

const model = await loadTensorflowModel(
  { url: 'https://example.com/large_model.tflite' },
  'default',
  (progress) => {
    if (progress === -1) {
      console.log('Downloading… (size unknown)')
    } else {
      console.log(`Downloading: ${Math.round(progress * 100)}%`)
    }
  }
)
```

> `progress` is in `[0, 1]`. Receives `-1` when the server does not send a `Content-Length` header.

---

## Model Caching

Models are automatically cached in native memory by `URL + delegate`.  
Calling `loadTensorflowModel` twice with the same arguments returns the **same model** instantly:

```ts
const m1 = await loadTensorflowModel(require('./model.tflite'))
const m2 = await loadTensorflowModel(require('./model.tflite')) // cache hit — instant
// m1 and m2 share the same underlying TfLiteInterpreter
```

To eagerly free all cached models (e.g. on screen unmount):

```ts
import { clearTensorflowModelCache } from 'react-native-nitro-tflite'

clearTensorflowModelCache()
```

---

## Dynamic Input Shape

For models that support dynamic tensor shapes, call `reshapeInput` before inference:

```ts
const model = await loadTensorflowModel(require('./dynamic_model.tflite'))

// Switch to 640×640 input at runtime
model.reshapeInput(0, [1, 640, 640, 3])

const output = model.runSync([new Float32Array(1 * 640 * 640 * 3)])
```

> After `reshapeInput`, both input and output tensor sizes are re-allocated automatically.

---

## Delegates

| Delegate      | Platform | Description                              | Setup                                    |
| ------------- | -------- | ---------------------------------------- | ---------------------------------------- |
| `default`     | Both     | CPU inference (always available)         | —                                        |
| `core-ml`     | iOS      | Apple CoreML acceleration                | `$EnableCoreMLDelegate = true` in Podfile |
| `metal`       | iOS      | Metal GPU acceleration                   | `$EnableMetalDelegate = true` in Podfile  |
| `nnapi`       | Android  | Android Neural Networks API              | —                                        |
| `android-gpu` | Android  | Android GPU delegate                     | —                                        |

### Enabling CoreML (iOS)

```ruby
# ios/Podfile
$EnableCoreMLDelegate = true

use_frameworks! ...
```

### Enabling Metal (iOS)

```ruby
# ios/Podfile
$EnableMetalDelegate = true

use_frameworks! ...
```

Then re-run `pod install`.

---

## Metro Configuration

Add `tflite` as an asset extension in `metro.config.js`:

```js
const { getDefaultConfig } = require('@react-native/metro-config')

const config = getDefaultConfig(__dirname)
config.resolver.assetExts.push('tflite')

module.exports = config
```

---

## Architecture

This library uses [Nitro Modules](https://nitro.margelo.com) with manual C++ `HybridObject` implementation:

- **`HybridTfliteModelFactory`** — Singleton factory; loads models on a background thread, manages the cache, handles delegate selection
- **`HybridTfliteModel`** — Wraps `TfLiteInterpreter`; exposes `run` / `runSync` / `reshapeInput` / `inputs` / `outputs` / `delegate`
- **`TfliteModelHostObject`** — A `jsi::HostObject` wrapper that makes the model usable across worklet runtimes (VisionCamera frame processors)

```
JS:  loadTensorflowModel(source, delegate, onProgress?)
       ↓
     resolveSourceUri()  →  URI string
       ↓
     factory.loadModel(uri, delegate, onProgress)
       ↓  [cache hit → resolve immediately]
       ↓  [cache miss → background thread]
C++: HybridTfliteModelFactory::loadModelRaw()
       ↓
     fetchURL(uri, onProgress)   ← platform-specific (iOS: NSURLSession, Android: OkHttp)
       ↓
     TfLiteModelCreate() + TfLiteInterpreterCreate()
       ↓
     HybridTfliteModel  →  TfliteModelHostObject  →  JS Promise resolved
       ↓
     _modelCache[key] = weak_ptr<HybridTfliteModel>
```

---

## License

MIT
