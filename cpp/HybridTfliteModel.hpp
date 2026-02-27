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
#include <memory>
#include <string>
#include <unordered_map>

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

public:
  explicit HybridTfliteModel(TfLiteModel* model,
                               TfLiteInterpreter* interpreter,
                               Buffer modelData,
                               Delegate delegate);
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

  std::unordered_map<std::string, std::shared_ptr<TypedArrayBase>> _outputBuffers;

  static constexpr auto TAG = "TfliteModel";
};
