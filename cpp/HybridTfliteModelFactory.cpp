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
#endif

using namespace facebook;

std::shared_ptr<HybridTfliteModelFactory> HybridTfliteModelFactory::_instance = nullptr;

HybridTfliteModelFactory::HybridTfliteModelFactory()
    : HybridObject(TAG) {}

void HybridTfliteModelFactory::setFetchURLFunc(FetchURLFunc fetchFunc) {
  _fetchURL = std::move(fetchFunc);
}

std::shared_ptr<HybridTfliteModelFactory> HybridTfliteModelFactory::getOrCreate() {
  if (_instance == nullptr) {
    _instance = std::make_shared<HybridTfliteModelFactory>();
  }
  return _instance;
}

HybridTfliteModel::Delegate HybridTfliteModelFactory::parseDelegateString(const std::string& delegate) {
  if (delegate == "core-ml") {
    return HybridTfliteModel::Delegate::CoreML;
  } else if (delegate == "metal") {
    return HybridTfliteModel::Delegate::Metal;
  } else if (delegate == "nnapi") {
    return HybridTfliteModel::Delegate::NnApi;
  } else if (delegate == "android-gpu") {
    return HybridTfliteModel::Delegate::AndroidGPU;
  } else {
    return HybridTfliteModel::Delegate::Default;
  }
}

size_t HybridTfliteModelFactory::getExternalMemorySize() noexcept {
  return 0;
}

jsi::Value HybridTfliteModelFactory::loadModelRaw(jsi::Runtime& runtime,
                                                   const jsi::Value& thisValue,
                                                   const jsi::Value* args,
                                                   size_t count) {
  if (count < 1 || !args[0].isString()) {
    throw jsi::JSError(runtime, "TFLite: loadModel requires a path string as first argument!");
  }

  auto modelPath = args[0].asString(runtime).utf8(runtime);

  // Parse delegate
  HybridTfliteModel::Delegate delegateType = HybridTfliteModel::Delegate::Default;
  if (count > 1 && args[1].isString()) {
    auto delegateStr = args[1].asString(runtime).utf8(runtime);
    delegateType = parseDelegateString(delegateStr);
  }

  auto fetchURL = _fetchURL;
  if (!fetchURL) {
    throw jsi::JSError(runtime, "TFLite: Platform fetch function not set! "
                                "Make sure the native module is properly initialized.");
  }

  // Get the JS thread dispatcher so we can safely call resolve/reject from
  // the background thread.
  auto dispatcher = margelo::nitro::Dispatcher::getRuntimeGlobalDispatcher(runtime);

  auto promise = mrousavy::Promise::createPromise(runtime, [=](std::shared_ptr<mrousavy::Promise> promise) {
    std::thread([=]() {
      try {
        auto start = std::chrono::steady_clock::now();

        LOGI("Loading model from: %s", modelPath.c_str());

        // Fetch model from URL (JS bundle or file)
        Buffer buffer = fetchURL(modelPath);
        LOGI("Fetched %zu bytes for model", buffer.size);

        // Load Model into TensorFlow Lite
        auto model = TfLiteModelCreate(buffer.data, buffer.size);
        if (model == nullptr) {
          auto msg = std::string("Failed to load model from \"") + modelPath + "\"!";
          dispatcher->runAsync([promise, msg = std::move(msg)]() {
            promise->reject(msg);
          });
          return;
        }

        // Create TensorFlow Interpreter with options
        auto options = TfLiteInterpreterOptionsCreate();

        switch (delegateType) {
          case HybridTfliteModel::Delegate::CoreML: {
#if FAST_TFLITE_ENABLE_CORE_ML
            TfLiteCoreMlDelegateOptions delegateOptions;
            auto delegate = TfLiteCoreMlDelegateCreate(&delegateOptions);
            TfLiteInterpreterOptionsAddDelegate(options, delegate);
            break;
#else
            dispatcher->runAsync([promise]() {
              promise->reject("CoreML Delegate is not enabled! Set $EnableCoreMLDelegate to true in Podfile and rebuild.");
            });
            return;
#endif
          }
          case HybridTfliteModel::Delegate::Metal: {
            dispatcher->runAsync([promise]() {
              promise->reject("Metal Delegate is not supported!");
            });
            return;
          }
#ifdef ANDROID
          case HybridTfliteModel::Delegate::NnApi: {
            TfLiteNnapiDelegateOptions delegateOptions = TfLiteNnapiDelegateOptionsDefault();
            auto delegate = TfLiteNnapiDelegateCreate(&delegateOptions);
            TfLiteInterpreterOptionsAddDelegate(options, delegate);
            break;
          }
          case HybridTfliteModel::Delegate::AndroidGPU: {
            TfLiteGpuDelegateOptionsV2 delegateOptions = TfLiteGpuDelegateOptionsV2Default();
            auto delegate = TfLiteGpuDelegateV2Create(&delegateOptions);
            TfLiteInterpreterOptionsAddDelegate(options, delegate);
            break;
          }
#else
          case HybridTfliteModel::Delegate::NnApi: {
            dispatcher->runAsync([promise]() {
              promise->reject("NNAPI Delegate is only supported on Android!");
            });
            return;
          }
          case HybridTfliteModel::Delegate::AndroidGPU: {
            dispatcher->runAsync([promise]() {
              promise->reject("Android GPU Delegate is only supported on Android!");
            });
            return;
          }
#endif
          default: {
            // Use default CPU delegate
            break;
          }
        }

        auto interpreter = TfLiteInterpreterCreate(model, options);
        if (interpreter == nullptr) {
          auto msg = std::string("Failed to create TFLite interpreter from model \"") + modelPath + "\"!";
          dispatcher->runAsync([promise, msg = std::move(msg)]() {
            promise->reject(msg);
          });
          return;
        }

        // Create the HybridTfliteModel
        auto hybridModel = std::make_shared<HybridTfliteModel>(model, interpreter, buffer, delegateType);

        // Resolve the promise on the JS thread with a HostObject wrapper.
        // HostObject (unlike NitroModules HybridObject) works across worklet
        // runtimes because it doesn't rely on jsi::NativeState.
        dispatcher->runAsync([promise, hybridModel]() {
          auto hostObject = std::make_shared<TfliteModelHostObject>(hybridModel);
          auto result = jsi::Object::createFromHostObject(promise->runtime, hostObject);
          promise->resolve(jsi::Value(std::move(result)));
        });

        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // Model loaded successfully
        (void)ms;

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

void HybridTfliteModelFactory::loadHybridMethods() {
  // Register base methods (toString, name, equals, dispose)
  HybridObject::loadHybridMethods();

  // Register our loadModel method
  registerHybrids(this, [](Prototype& prototype) {
    prototype.registerRawHybridMethod("loadModel", 2, &HybridTfliteModelFactory::loadModelRaw);
  });
}
