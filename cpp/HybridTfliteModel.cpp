//
//  HybridTfliteModel.cpp
//  react-native-nitro-tflite
//
//  TFLite Model HybridObject implementation.
//

#include "HybridTfliteModel.hpp"
#include "TensorHelpers.h"
#include "jsi/Promise.h"
#include "jsi/TypedArray.h"
#include <string>

#ifdef ANDROID
#include <tflite/c/c_api.h>
#else
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#endif

using namespace facebook;
using namespace mrousavy;

static std::string tfLiteStatusToString(TfLiteStatus status) {
  switch (status) {
    case kTfLiteOk:
      return "ok";
    case kTfLiteError:
      return "error";
    case kTfLiteDelegateError:
      return "delegate-error";
    case kTfLiteApplicationError:
      return "application-error";
    case kTfLiteDelegateDataNotFound:
      return "delegate-data-not-found";
    case kTfLiteDelegateDataWriteError:
      return "delegate-data-write-error";
    case kTfLiteDelegateDataReadError:
      return "delegate-data-read-error";
    case kTfLiteUnresolvedOps:
      return "unresolved-ops";
    case kTfLiteCancelled:
      return "cancelled";
    default:
      return "unknown";
  }
}

HybridTfliteModel::HybridTfliteModel(TfLiteModel* model,
                                       TfLiteInterpreter* interpreter,
                                       Buffer modelData,
                                       Delegate delegate)
    : HybridObject(TAG), _modelPtr(model), _interpreter(interpreter), _delegate(delegate), _model(modelData) {
  // Allocate memory for the model's input/output TFLTensors.
  TfLiteStatus status = TfLiteInterpreterAllocateTensors(_interpreter);
  if (status != kTfLiteOk) {
    throw std::runtime_error(
        "TFLite: Failed to allocate memory for input/output tensors! Status: " +
        tfLiteStatusToString(status));
  }
}

HybridTfliteModel::~HybridTfliteModel() {
  if (_model.data != nullptr) {
    free(_model.data);
    _model.data = nullptr;
    _model.size = 0;
  }
  if (_interpreter != nullptr) {
    TfLiteInterpreterDelete(_interpreter);
    _interpreter = nullptr;
  }
  if (_modelPtr != nullptr) {
    TfLiteModelDelete(_modelPtr);
    _modelPtr = nullptr;
  }
}

size_t HybridTfliteModel::getExternalMemorySize() noexcept {
  return _model.size;
}

std::string HybridTfliteModel::delegateToString() const {
  switch (_delegate) {
    case Delegate::Default:
      return "default";
    case Delegate::CoreML:
      return "core-ml";
    case Delegate::Metal:
      return "metal";
    case Delegate::NnApi:
      return "nnapi";
    case Delegate::AndroidGPU:
      return "android-gpu";
    default:
      return "default";
  }
}

std::shared_ptr<TypedArrayBase>
HybridTfliteModel::getOutputArrayForTensor(jsi::Runtime& runtime, const TfLiteTensor* tensor) {
  auto name = std::string(TfLiteTensorName(tensor));
  if (_outputBuffers.find(name) == _outputBuffers.end()) {
    _outputBuffers[name] =
        std::make_shared<TypedArrayBase>(TensorHelpers::createJSBufferForTensor(runtime, tensor));
  }
  return _outputBuffers[name];
}

void HybridTfliteModel::copyInputBuffers(jsi::Runtime& runtime, jsi::Object inputValues) {
#if DEBUG
  if (!inputValues.isArray(runtime)) {
    throw jsi::JSError(runtime,
                       "TFLite: Input Values must be an array, one item for each input tensor!");
  }
#endif

  jsi::Array array = inputValues.asArray(runtime);
  size_t count = array.size(runtime);
  if (count != TfLiteInterpreterGetInputTensorCount(_interpreter)) {
    throw jsi::JSError(runtime,
                       "TFLite: Input Values have different size than there are input tensors!");
  }

  for (size_t i = 0; i < count; i++) {
    TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(_interpreter, i);
    jsi::Object object = array.getValueAtIndex(runtime, i).asObject(runtime);

#if DEBUG
    if (!isTypedArray(runtime, object)) {
      throw jsi::JSError(
          runtime,
          "TFLite: Input value is not a TypedArray! (Uint8Array, Float32Array, etc.)");
    }
#endif

    TypedArrayBase inputBuffer = getTypedArray(runtime, std::move(object));
    TensorHelpers::updateTensorFromJSBuffer(runtime, tensor, inputBuffer);
  }
}

jsi::Value HybridTfliteModel::copyOutputBuffers(jsi::Runtime& runtime) {
  int outputTensorsCount = TfLiteInterpreterGetOutputTensorCount(_interpreter);
  jsi::Array result(runtime, outputTensorsCount);
  for (size_t i = 0; i < outputTensorsCount; i++) {
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(_interpreter, i);
    auto outputBuffer = getOutputArrayForTensor(runtime, outputTensor);
    TensorHelpers::updateJSBufferFromTensor(runtime, *outputBuffer, outputTensor);
    result.setValueAtIndex(runtime, i, *outputBuffer);
  }
  return result;
}

void HybridTfliteModel::runInference() {
  TfLiteStatus status = TfLiteInterpreterInvoke(_interpreter);
  if (status != kTfLiteOk) {
    throw std::runtime_error("TFLite: Failed to run TFLite Model! Status: " +
                             tfLiteStatusToString(status));
  }
}

// Raw JSI methods

jsi::Value HybridTfliteModel::runSyncRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                                          const jsi::Value* args, size_t count) {
  // 1. Copy input data
  copyInputBuffers(runtime, args[0].asObject(runtime));
  // 2. Run inference
  runInference();
  // 3. Copy output data
  return copyOutputBuffers(runtime);
}

jsi::Value HybridTfliteModel::runAsyncRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                                           const jsi::Value* args, size_t count) {
  // In Nitro Modules, HybridObjects can be called from any worklet/thread.
  // Without a CallInvoker, we run inference synchronously and wrap in a Promise.
  // This is safe because Nitro ensures the runtime is valid on the calling thread.
  auto promise = mrousavy::Promise::createPromise(runtime, [this, &runtime, args](std::shared_ptr<mrousavy::Promise> promise) {
    try {
      // 1. Copy input data
      copyInputBuffers(runtime, args[0].asObject(runtime));
      // 2. Run inference
      runInference();
      // 3. Copy output data and resolve
      auto result = copyOutputBuffers(runtime);
      promise->resolve(std::move(result));
    } catch (std::exception& error) {
      promise->reject(error.what());
    }
  });
  return promise;
}

jsi::Value HybridTfliteModel::getInputsRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                                             const jsi::Value* args, size_t count) {
  int size = TfLiteInterpreterGetInputTensorCount(_interpreter);
  jsi::Array tensors(runtime, size);
  for (size_t i = 0; i < size; i++) {
    TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(_interpreter, i);
    if (tensor == nullptr) {
      throw jsi::JSError(runtime,
                         "TFLite: Failed to get input tensor " + std::to_string(i) + "!");
    }
    jsi::Object object = TensorHelpers::tensorToJSObject(runtime, tensor);
    tensors.setValueAtIndex(runtime, i, object);
  }
  return tensors;
}

jsi::Value HybridTfliteModel::getOutputsRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                                              const jsi::Value* args, size_t count) {
  int size = TfLiteInterpreterGetOutputTensorCount(_interpreter);
  jsi::Array tensors(runtime, size);
  for (size_t i = 0; i < size; i++) {
    const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(_interpreter, i);
    if (tensor == nullptr) {
      throw jsi::JSError(runtime,
                         "TFLite: Failed to get output tensor " + std::to_string(i) + "!");
    }
    jsi::Object object = TensorHelpers::tensorToJSObject(runtime, tensor);
    tensors.setValueAtIndex(runtime, i, object);
  }
  return tensors;
}

jsi::Value HybridTfliteModel::getDelegateRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                                              const jsi::Value* args, size_t count) {
  return jsi::String::createFromUtf8(runtime, delegateToString());
}

void HybridTfliteModel::loadHybridMethods() {
  // Register base methods (toString, name, equals, dispose)
  HybridObject::loadHybridMethods();

  // Register our raw JSI methods
  registerHybrids(this, [](Prototype& prototype) {
    prototype.registerRawHybridMethod("runSync", 1, &HybridTfliteModel::runSyncRaw);
    prototype.registerRawHybridMethod("run", 1, &HybridTfliteModel::runAsyncRaw);
    prototype.registerRawHybridMethod("inputs", 0, &HybridTfliteModel::getInputsRaw);
    prototype.registerRawHybridMethod("outputs", 0, &HybridTfliteModel::getOutputsRaw);
    prototype.registerRawHybridMethod("delegate", 0, &HybridTfliteModel::getDelegateRaw);
  });
}
