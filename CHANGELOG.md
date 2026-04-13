# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-13

### Added

- **Model caching** — `loadTensorflowModel` and the native factory now maintain a
  weak-reference in-memory cache keyed by `URL + delegate`. Calling the function
  twice with the same arguments returns the existing model instantly without
  re-reading or re-allocating tensors. ([#cache])
  - A new JS helper `clearTensorflowModelCache()` lets you eagerly free all
    cached entries.
  - Cache entries expire automatically when the model object is garbage-collected.

- **Download progress callback** — `loadTensorflowModel` now accepts an optional
  third argument `onProgress?: (progress: number) => void`. ([#progress])
  - `progress` is in the range `[0, 1]`.
  - `progress === -1` when the server does not send a `Content-Length` header.
  - iOS: implemented with `NSURLSession` streaming (replaces the blocking
    `dataWithContentsOfURL:` call — also fixes a main-thread stall bug).
  - Android: implemented with OkHttp chunked streaming.
  - Local `file://` and bundled assets report `1.0` (100 %) on completion.

- **Metal GPU delegate** (iOS) — Pass `'metal'` as the delegate to offload
  inference to the GPU via Apple's Metal API. ([#metal])
  - Enable the TFLite Metal pod by adding `$EnableMetalDelegate = true` in your
    `Podfile` before `use_frameworks!`, then run `pod install`.
  - Requires `TensorFlowLiteCMetal` pod (automatically added when the flag is set).

- **Dynamic input shape** — New `model.reshapeInput(inputIndex, shape)` method
  for models that support dynamic tensor shapes. ([#reshape])
  - Calls `TfLiteInterpreterResizeInputTensor` + `TfLiteInterpreterAllocateTensors`
    under the hood, and clears the output buffer cache so sizes are recomputed.
  - Exposed on both the Nitro `HybridObject` path and the `HostObject` path
    (used in cross-runtime / VisionCamera frame-processor contexts).

- `clearTensorflowModelCache()` top-level export.
- `ModelLoadProgress` type export.
- `ModelSource` type export.

### Changed

- **`useTensorflowModel`**: The `useEffect` dependency on `source` is now keyed
  by the underlying URI string (`sourceKey`) instead of the object reference.
  This prevents infinite reload loops when callers inline `{ url }` objects
  (e.g. `useTensorflowModel({ url: '...' })`).
- **`loadTensorflowModel`**: Source resolution logic extracted to a standalone
  `resolveSourceUri` helper.
- **`TfliteModelFactory.loadModel`** type signature: `delegate` is now typed as
  `TensorflowModelDelegate` (was `string`) — oops-prone string values are now
  caught at compile time.
- Removed `'float16'` from the `Tensor.dataType` union — it was present in the
  TypeScript types but not handled in C++, leading to a guaranteed runtime error.
- **Singleton**: `HybridTfliteModelFactory::getOrCreate()` now uses
  `std::call_once` for thread-safe initialisation.

### Fixed

- **Memory leak**: `TfLiteInterpreterOptionsCreate()` result was never deleted.
  `TfLiteInterpreterOptionsDelete(options)` is now called unconditionally after
  the interpreter is created (or when creation fails).
- **iOS blocking network call**: `[NSData dataWithContentsOfURL:]` blocked the
  calling thread for HTTP URLs. The installer now uses `NSURLSession`.
- **Podspec source URL**: was `github.com/user/react-native-nitro-tflite`
  (placeholder). Fixed to `github.com/dodokw/react-native-nitro-tflite`.

---

## [0.0.1] — 2026-04-13

### Added

- Initial release of `react-native-nitro-tflite`.
- HybridObject-based TFLite wrapper powered by Nitro Modules.
- `loadTensorflowModel()` and `useTensorflowModel()` API, identical to
  `react-native-fast-tflite` for easy migration.
- `run()` (async) and `runSync()` for inference.
- `inputs` / `outputs` tensor metadata.
- Delegate support: `default` (CPU), `core-ml`, `nnapi`, `android-gpu`.
- Works from any thread/worklet runtime via `jsi::HostObject` wrapping.
- iOS: `TensorFlowLiteC` 2.17.0.
- Android: `litert` 1.4.0 + `litert-gpu` 1.4.0 via Gradle.
