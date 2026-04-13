//
//  HybridTfliteModel.hpp
//  react-native-nitro-tflite
//
//  TFLite Model as a Nitro HybridObject.
//  Wraps a TfLiteInterpreter and exposes run/runSync/inputs/outputs/delegate.
//

#pragma once

#include <NitroModules/HybridObject.hpp>
#include "jsi/TypedArray.h"
#include <jsi/jsi.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ANDROID
#include <tflite/c/c_api.h>
#else
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#endif

using namespace facebook;
using namespace margelo::nitro;
using namespace mrousavy;

struct Buffer {
  void* data;
  size_t size;
};

class HybridTfliteModel : public HybridObject {
public:
  // TFL Delegate Type
  enum Delegate { Default, Metal, CoreML, NnApi, AndroidGPU };

  // DelegateDeleter: platform-specific cleanup for the TFLite delegate.
  // Each delegate API has its own delete function.
  using DelegateDeleter = std::function<void()>;

public:
  explicit HybridTfliteModel(TfLiteModel* model,
                               TfLiteInterpreter* interpreter,
                               Buffer modelData,
                               Delegate delegate,
                               DelegateDeleter delegateDeleter = nullptr);
  ~HybridTfliteModel() override;

  // HybridObject overrides
  size_t getExternalMemorySize() noexcept override;

protected:
  void loadHybridMethods() override;

public:
  // Core methods exposed to JS
  jsi::Value runSyncRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                        const jsi::Value* args, size_t count);
  jsi::Value runAsyncRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                         const jsi::Value* args, size_t count);
  jsi::Value getInputsRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                          const jsi::Value* args, size_t count);
  jsi::Value getOutputsRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                           const jsi::Value* args, size_t count);
  jsi::Value getDelegateRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                            const jsi::Value* args, size_t count);
  /**
   * Dynamically resize an input tensor.
   * args[0]: inputIndex (number), args[1]: newShape (number[])
   * Calls TfLiteInterpreterResizeInputTensor + TfLiteInterpreterAllocateTensors.
   */
  jsi::Value reshapeInputRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                              const jsi::Value* args, size_t count);

  // Internal helpers
  void copyInputBuffers(jsi::Runtime& runtime, jsi::Object inputValues);
  void runInference();
  jsi::Value copyOutputBuffers(jsi::Runtime& runtime);

  std::shared_ptr<TypedArrayBase> getOutputArrayForTensor(jsi::Runtime& runtime,
                                                          const TfLiteTensor* tensor);

  std::string delegateToString() const;

private:
  TfLiteModel* _modelPtr = nullptr;
  TfLiteInterpreter* _interpreter = nullptr;
  Delegate _delegate = Delegate::Default;
  Buffer _model;
  DelegateDeleter _delegateDeleter;  // cleans up the TFLite delegate on destruction

  std::unordered_map<std::string, std::shared_ptr<TypedArrayBase>> _outputBuffers;

  // Cached output result array — avoids allocating a new jsi::Array every inference.
  // Keyed by runtime pointer since the array is runtime-specific.
  std::unordered_map<uintptr_t, std::shared_ptr<jsi::Object>> _outputResultCache;
  int _cachedOutputCount = -1;

  static constexpr auto TAG = "TfliteModel";
};
