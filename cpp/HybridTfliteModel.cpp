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
                                       Delegate delegate,
                                       DelegateDeleter delegateDeleter)
    : HybridObject(TAG), _modelPtr(model), _interpreter(interpreter),
      _delegate(delegate), _model(modelData),
      _delegateDeleter(std::move(delegateDeleter)) {
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
  // Order matters: interpreter must be deleted before the delegate,
  // and the delegate before the model.
  if (_interpreter != nullptr) {
    TfLiteInterpreterDelete(_interpreter);
    _interpreter = nullptr;
  }
  // Delete the delegate (CoreML, Metal, NNAPI, GPU, etc.)
  if (_delegateDeleter) {
    _delegateDeleter();
    _delegateDeleter = nullptr;
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

  // Reuse a cached jsi::Array per runtime to avoid allocating a new one every
  // inference call.  The array size only changes after reshapeInput().
  auto runtimeKey = reinterpret_cast<uintptr_t>(&runtime);
  if (_cachedOutputCount != outputTensorsCount ||
      _outputResultCache.find(runtimeKey) == _outputResultCache.end()) {
    _outputResultCache[runtimeKey] =
        std::make_shared<jsi::Object>(jsi::Array(runtime, outputTensorsCount));
    _cachedOutputCount = outputTensorsCount;
  }
  auto& result = *_outputResultCache[runtimeKey];

  for (size_t i = 0; i < outputTensorsCount; i++) {
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(_interpreter, i);
    auto outputBuffer = getOutputArrayForTensor(runtime, outputTensor);
    TensorHelpers::updateJSBufferFromTensor(runtime, *outputBuffer, outputTensor);
    result.asArray(runtime).setValueAtIndex(runtime, i, *outputBuffer);
  }
  return jsi::Value(runtime, result);
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

jsi::Value HybridTfliteModel::runAsyncRaw(jsi::Runtime& runtime, const jsi::Value& /*thisValue*/,
                                           const jsi::Value* args, size_t count) {
  // Validate argument count up-front.
  if (count < 1) {
    throw jsi::JSError(runtime, "TFLite: run() requires at least one argument (input array)!");
  }

  // NOTE: Promise::createPromise executes the callback **synchronously** (inline),
  // so capturing `runtime` and `args` by reference is safe — they are guaranteed
  // to be alive for the duration of the lambda.
  auto promise = mrousavy::Promise::createPromise(runtime,
      [this, &runtime, args](std::shared_ptr<mrousavy::Promise> promise) {
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

jsi::Value HybridTfliteModel::reshapeInputRaw(jsi::Runtime& runtime,
                                               const jsi::Value& /*thisValue*/,
                                               const jsi::Value* args,
                                               size_t count) {
  if (count < 2 || !args[0].isNumber() || !args[1].isObject()) {
    throw jsi::JSError(runtime,
        "TFLite: reshapeInput(inputIndex: number, shape: number[]) expects two arguments!");
  }

  int inputIndex = static_cast<int>(args[0].asNumber());
  int inputCount = TfLiteInterpreterGetInputTensorCount(_interpreter);
  if (inputIndex < 0 || inputIndex >= inputCount) {
    throw jsi::JSError(runtime,
        "TFLite: inputIndex " + std::to_string(inputIndex) +
        " is out of range (model has " + std::to_string(inputCount) + " inputs)!");
  }

  jsi::Array shapeArray = args[1].asObject(runtime).asArray(runtime);
  size_t dims = shapeArray.size(runtime);
  if (dims == 0) {
    throw jsi::JSError(runtime, "TFLite: shape array must not be empty!");
  }

  std::vector<int> newShape(dims);
  for (size_t i = 0; i < dims; i++) {
    jsi::Value v = shapeArray.getValueAtIndex(runtime, i);
    if (!v.isNumber()) {
      throw jsi::JSError(runtime, "TFLite: all shape values must be numbers!");
    }
    newShape[i] = static_cast<int>(v.asNumber());
  }

  TfLiteStatus status = TfLiteInterpreterResizeInputTensor(
      _interpreter, inputIndex, newShape.data(), static_cast<int>(dims));
  if (status != kTfLiteOk) {
    throw jsi::JSError(runtime,
        "TFLite: TfLiteInterpreterResizeInputTensor failed for input " +
        std::to_string(inputIndex) + "!");
  }

  status = TfLiteInterpreterAllocateTensors(_interpreter);
  if (status != kTfLiteOk) {
    throw jsi::JSError(runtime,
        "TFLite: TfLiteInterpreterAllocateTensors failed after reshape!");
  }

  // Output buffer sizes may have changed — clear all caches so they are
  // re-created on the next copyOutputBuffers call.
  _outputBuffers.clear();
  _outputResultCache.clear();
  _cachedOutputCount = -1;

  return jsi::Value::undefined();
}

void HybridTfliteModel::loadHybridMethods() {
  // Register base methods (toString, name, equals, dispose)
  HybridObject::loadHybridMethods();

  // Register our raw JSI methods
  registerHybrids(this, [](Prototype& prototype) {
    prototype.registerRawHybridMethod("runSync",      1, &HybridTfliteModel::runSyncRaw);
    prototype.registerRawHybridMethod("run",          1, &HybridTfliteModel::runAsyncRaw);
    prototype.registerRawHybridMethod("inputs",       0, &HybridTfliteModel::getInputsRaw);
    prototype.registerRawHybridMethod("outputs",      0, &HybridTfliteModel::getOutputsRaw);
    prototype.registerRawHybridMethod("delegate",     0, &HybridTfliteModel::getDelegateRaw);
    prototype.registerRawHybridMethod("reshapeInput", 2, &HybridTfliteModel::reshapeInputRaw);
  });
}
