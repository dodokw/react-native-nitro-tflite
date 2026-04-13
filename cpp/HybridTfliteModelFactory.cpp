//
//  HybridTfliteModelFactory.cpp
//  react-native-nitro-tflite
//
//  Factory HybridObject implementation for loading TFLite models.
//

#include "HybridTfliteModelFactory.hpp"
#include "HybridTfliteModel.hpp"
#include "TfliteModelHostObject.hpp"
#include "jsi/Promise.h"
#include <NitroModules/HybridObjectRegistry.hpp>
#include <NitroModules/Dispatcher.hpp>
#include <chrono>
#include <mutex>
#include <thread>

#ifdef ANDROID
#include <android/log.h>
#define LOG_TAG "NitroTflite"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...)
#define LOGE(...)
#endif

#ifdef ANDROID
#include <tflite/c/c_api.h>
#include <tflite/delegates/gpu/delegate.h>
#include <tflite/delegates/nnapi/nnapi_delegate_c_api.h>
#else
#include <TensorFlowLiteC/TensorFlowLiteC.h>

#if FAST_TFLITE_ENABLE_CORE_ML
#include <TensorFlowLiteCCoreML/TensorFlowLiteCCoreML.h>
#endif

#if FAST_TFLITE_ENABLE_METAL
#include <TensorFlowLiteCMetal/TensorFlowLiteCMetal.h>
#endif
#endif

using namespace facebook;

// ────────────────────────────────────────────────────────────────
// Statics
// ────────────────────────────────────────────────────────────────
std::shared_ptr<HybridTfliteModelFactory> HybridTfliteModelFactory::_instance = nullptr;
std::once_flag HybridTfliteModelFactory::_instanceFlag;

// ────────────────────────────────────────────────────────────────
// Construction
// ────────────────────────────────────────────────────────────────
HybridTfliteModelFactory::HybridTfliteModelFactory()
    : HybridObject(TAG) {}

void HybridTfliteModelFactory::setFetchURLFunc(FetchURLFunc fetchFunc) {
  _fetchURL = std::move(fetchFunc);
}

std::shared_ptr<HybridTfliteModelFactory> HybridTfliteModelFactory::getOrCreate() {
  std::call_once(_instanceFlag, [] {
    _instance = std::make_shared<HybridTfliteModelFactory>();
  });
  return _instance;
}

// ────────────────────────────────────────────────────────────────
// Cache management
// ────────────────────────────────────────────────────────────────
void HybridTfliteModelFactory::clearModelCache() {
  std::lock_guard<std::mutex> lock(_cacheMutex);
  _modelCache.clear();
  LOGI("Model cache cleared");
}

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────
HybridTfliteModel::Delegate HybridTfliteModelFactory::parseDelegateString(const std::string& delegate) {
  if (delegate == "core-ml")    return HybridTfliteModel::Delegate::CoreML;
  if (delegate == "metal")      return HybridTfliteModel::Delegate::Metal;
  if (delegate == "nnapi")      return HybridTfliteModel::Delegate::NnApi;
  if (delegate == "android-gpu") return HybridTfliteModel::Delegate::AndroidGPU;
  return HybridTfliteModel::Delegate::Default;
}

size_t HybridTfliteModelFactory::getExternalMemorySize() noexcept {
  return 0;
}

// ────────────────────────────────────────────────────────────────
// clearCache (raw JSI)
// ────────────────────────────────────────────────────────────────
jsi::Value HybridTfliteModelFactory::clearCacheRaw(jsi::Runtime& runtime,
                                                    const jsi::Value& /*thisValue*/,
                                                    const jsi::Value* /*args*/,
                                                    size_t /*count*/) {
  clearModelCache();
  return jsi::Value::undefined();
}

// ────────────────────────────────────────────────────────────────
// loadModel (raw JSI)
// signature: loadModel(path: string, delegate: string, onProgress?: (p: number) => void)
// ────────────────────────────────────────────────────────────────
jsi::Value HybridTfliteModelFactory::loadModelRaw(jsi::Runtime& runtime,
                                                   const jsi::Value& /*thisValue*/,
                                                   const jsi::Value* args,
                                                   size_t count) {
  if (count < 1 || !args[0].isString()) {
    throw jsi::JSError(runtime, "TFLite: loadModel requires a path string as first argument!");
  }

  auto modelPath = args[0].asString(runtime).utf8(runtime);

  // Parse delegate
  std::string delegateStr = "default";
  HybridTfliteModel::Delegate delegateType = HybridTfliteModel::Delegate::Default;
  if (count > 1 && args[1].isString()) {
    delegateStr = args[1].asString(runtime).utf8(runtime);
    delegateType = parseDelegateString(delegateStr);
  }

  // Cache key combines URL + delegate so the same model with different
  // delegates is treated as distinct entries.
  std::string cacheKey = modelPath + ":" + delegateStr;

  auto fetchURL = _fetchURL;
  if (!fetchURL) {
    throw jsi::JSError(runtime, "TFLite: Platform fetch function not set! "
                                "Make sure the native module is properly initialized.");
  }

  // JS-thread dispatcher so we can safely call resolve/reject from background.
  auto dispatcher = margelo::nitro::Dispatcher::getRuntimeGlobalDispatcher(runtime);

  // ── Extract optional onProgress callback ────────────────────────────────
  // Store the raw runtime pointer so we can call back on the JS thread.
  // The runtime outlives the loading operation in all normal RN lifecycles.
  jsi::Runtime* runtimePtr = &runtime;
  std::shared_ptr<jsi::Function> progressFn;
  if (count > 2 && args[2].isObject()) {
    auto obj = args[2].asObject(runtime);
    if (obj.isFunction(runtime)) {
      progressFn = std::make_shared<jsi::Function>(obj.asFunction(runtime));
    }
  }

  // Build a C++ progress callback that marshals calls to the JS thread.
  ProgressCallback cppProgress = nullptr;
  if (progressFn) {
    cppProgress = [dispatcher, progressFn, runtimePtr](double progress) {
      dispatcher->runAsync([progressFn, progress, runtimePtr]() {
        progressFn->call(*runtimePtr, jsi::Value(progress));
      });
    };
  }

  // ── Check in-memory cache first (on JS thread, no async needed) ─────────
  {
    std::lock_guard<std::mutex> lock(_cacheMutex);
    auto it = _modelCache.find(cacheKey);
    if (it != _modelCache.end()) {
      auto cached = it->second.lock();
      if (cached) {
        LOGI("Cache hit for model: %s [%s]", modelPath.c_str(), delegateStr.c_str());
        // Fire progress at 100% to keep the caller consistent.
        if (cppProgress) cppProgress(1.0);
        return mrousavy::Promise::createPromise(runtime,
            [dispatcher, cached](std::shared_ptr<mrousavy::Promise> promise) {
              dispatcher->runAsync([promise, cached]() {
                auto hostObject = std::make_shared<TfliteModelHostObject>(cached);
                auto result = jsi::Object::createFromHostObject(promise->runtime, hostObject);
                promise->resolve(jsi::Value(std::move(result)));
              });
            });
      } else {
        // Stale entry — weak_ptr expired, remove it.
        _modelCache.erase(it);
      }
    }
  }

  // ── Load on a background thread ─────────────────────────────────────────
  auto promise = mrousavy::Promise::createPromise(runtime,
      [this, fetchURL, dispatcher, modelPath, delegateStr, delegateType, cacheKey, cppProgress]
      (std::shared_ptr<mrousavy::Promise> promise) {
        std::thread([this, fetchURL, dispatcher, modelPath, delegateStr, delegateType,
                     cacheKey, cppProgress, promise]() {
          try {
            auto start = std::chrono::steady_clock::now();
            LOGI("Loading model from: %s", modelPath.c_str());

            // ── Fetch bytes (platform-specific; progress forwarded) ────────
            Buffer buffer = fetchURL(modelPath, cppProgress);
            LOGI("Fetched %zu bytes for model", buffer.size);

            // Fire 100% if we haven't already.
            if (cppProgress) cppProgress(1.0);

            // ── Create TFLite model ───────────────────────────────────────
            auto model = TfLiteModelCreate(buffer.data, buffer.size);
            if (model == nullptr) {
              auto msg = std::string("Failed to load model from \"") + modelPath + "\"!";
              dispatcher->runAsync([promise, msg = std::move(msg)]() {
                promise->reject(msg);
              });
              return;
            }

            // ── Configure interpreter options ─────────────────────────────
            auto options = TfLiteInterpreterOptionsCreate();

            // Delegate cleanup function — called by ~HybridTfliteModel after
            // the interpreter is deleted. Each delegate API has its own
            // destruction function, so we capture it in a lambda.
            HybridTfliteModel::DelegateDeleter delegateDeleter = nullptr;

            switch (delegateType) {
              case HybridTfliteModel::Delegate::CoreML: {
#if FAST_TFLITE_ENABLE_CORE_ML
                TfLiteCoreMlDelegateOptions delegateOptions;
                auto delegate = TfLiteCoreMlDelegateCreate(&delegateOptions);
                TfLiteInterpreterOptionsAddDelegate(options, delegate);
                delegateDeleter = [delegate]() { TfLiteCoreMlDelegateDelete(delegate); };
                break;
#else
                TfLiteInterpreterOptionsDelete(options);
                TfLiteModelDelete(model);
                free(buffer.data);
                dispatcher->runAsync([promise]() {
                  promise->reject("CoreML Delegate is not enabled! "
                                  "Set $EnableCoreMLDelegate to true in Podfile and rebuild.");
                });
                return;
#endif
              }
              case HybridTfliteModel::Delegate::Metal: {
#if FAST_TFLITE_ENABLE_METAL
                auto metalDelegate = TFLGpuDelegateCreate(nil);
                TfLiteInterpreterOptionsAddDelegate(options, metalDelegate);
                delegateDeleter = [metalDelegate]() { TFLGpuDelegateDelete(metalDelegate); };
                break;
#else
                TfLiteInterpreterOptionsDelete(options);
                TfLiteModelDelete(model);
                free(buffer.data);
                dispatcher->runAsync([promise]() {
                  promise->reject("Metal Delegate is not enabled! "
                                  "Set $EnableMetalDelegate to true in Podfile and rebuild.");
                });
                return;
#endif
              }
#ifdef ANDROID
              case HybridTfliteModel::Delegate::NnApi: {
                TfLiteNnapiDelegateOptions delegateOptions = TfLiteNnapiDelegateOptionsDefault();
                auto delegate = TfLiteNnapiDelegateCreate(&delegateOptions);
                TfLiteInterpreterOptionsAddDelegate(options, delegate);
                delegateDeleter = [delegate]() { TfLiteNnapiDelegateDelete(delegate); };
                break;
              }
              case HybridTfliteModel::Delegate::AndroidGPU: {
                TfLiteGpuDelegateOptionsV2 delegateOptions = TfLiteGpuDelegateOptionsV2Default();
                auto delegate = TfLiteGpuDelegateV2Create(&delegateOptions);
                TfLiteInterpreterOptionsAddDelegate(options, delegate);
                delegateDeleter = [delegate]() { TfLiteGpuDelegateV2Delete(delegate); };
                break;
              }
#else
              case HybridTfliteModel::Delegate::NnApi: {
                TfLiteInterpreterOptionsDelete(options);
                TfLiteModelDelete(model);
                free(buffer.data);
                dispatcher->runAsync([promise]() {
                  promise->reject("NNAPI Delegate is only supported on Android!");
                });
                return;
              }
              case HybridTfliteModel::Delegate::AndroidGPU: {
                TfLiteInterpreterOptionsDelete(options);
                TfLiteModelDelete(model);
                free(buffer.data);
                dispatcher->runAsync([promise]() {
                  promise->reject("Android GPU Delegate is only supported on Android!");
                });
                return;
              }
#endif
              default:
                break; // CPU (default) — no delegate needed
            }

            // ── Create interpreter ────────────────────────────────────────
            auto interpreter = TfLiteInterpreterCreate(model, options);
            TfLiteInterpreterOptionsDelete(options); // ← always release options
            if (interpreter == nullptr) {
              // Clean up delegate if it was created
              if (delegateDeleter) delegateDeleter();
              TfLiteModelDelete(model);
              free(buffer.data);
              auto msg = std::string("Failed to create TFLite interpreter from model \"") + modelPath + "\"!";
              dispatcher->runAsync([promise, msg = std::move(msg)]() {
                promise->reject(msg);
              });
              return;
            }

            // ── Build HybridTfliteModel ───────────────────────────────────
            auto hybridModel = std::make_shared<HybridTfliteModel>(
                model, interpreter, buffer, delegateType, std::move(delegateDeleter));

            // ── Store in cache (weak_ptr — model can still be GC'd) ───────
            {
              std::lock_guard<std::mutex> lock(_cacheMutex);
              _modelCache[cacheKey] = hybridModel;
            }

            auto end = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            LOGI("Model loaded in %lldms: %s [%s]", ms, modelPath.c_str(), delegateStr.c_str());

            // ── Resolve promise on JS thread ──────────────────────────────
            dispatcher->runAsync([promise, hybridModel]() {
              auto hostObject = std::make_shared<TfliteModelHostObject>(hybridModel);
              auto result = jsi::Object::createFromHostObject(promise->runtime, hostObject);
              promise->resolve(jsi::Value(std::move(result)));
            });

          } catch (const std::exception& error) {
            std::string message = error.what();
            LOGE("Model loading failed (std::exception): %s", message.c_str());
            dispatcher->runAsync([promise, message = std::move(message)]() {
              promise->reject(message);
            });
          } catch (...) {
            LOGE("Model loading failed with unknown exception");
            dispatcher->runAsync([promise]() {
              promise->reject("Unknown error occurred while loading TFLite model");
            });
          }
        }).detach();
      });

  return promise;
}

// ────────────────────────────────────────────────────────────────
// loadHybridMethods
// ────────────────────────────────────────────────────────────────
void HybridTfliteModelFactory::loadHybridMethods() {
  HybridObject::loadHybridMethods();

  registerHybrids(this, [](Prototype& prototype) {
    prototype.registerRawHybridMethod("loadModel",  3, &HybridTfliteModelFactory::loadModelRaw);
    prototype.registerRawHybridMethod("clearCache", 0, &HybridTfliteModelFactory::clearCacheRaw);
  });
}
