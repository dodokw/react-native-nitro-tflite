//
//  TfliteModelHostObject.hpp
//  react-native-nitro-tflite
//
//  A jsi::HostObject wrapper around HybridTfliteModel that is compatible
//  with cross-runtime sharing (e.g., worklet runtimes in VisionCamera).
//
//  NitroModules' HybridObject uses jsi::NativeState which is runtime-specific.
//  jsi::HostObject works across runtimes because they are shared by reference.
//

#pragma once

#include <jsi/jsi.h>
#include "HybridTfliteModel.hpp"
#include <memory>
#include <string>

using namespace facebook;

class TfliteModelHostObject : public jsi::HostObject {
public:
  explicit TfliteModelHostObject(std::shared_ptr<HybridTfliteModel> model)
      : _model(std::move(model)) {}

  jsi::Value get(jsi::Runtime& rt, const jsi::PropNameID& propName) override {
    auto name = propName.utf8(rt);

    if (name == "runSync") {
      return jsi::Function::createFromHostFunction(
          rt, jsi::PropNameID::forAscii(rt, "runSync"), 1,
          [this](jsi::Runtime& runtime, const jsi::Value& thisVal,
                 const jsi::Value* args, size_t count) -> jsi::Value {
            return _model->runSyncRaw(runtime, thisVal, args, count);
          });
    }

    if (name == "run") {
      return jsi::Function::createFromHostFunction(
          rt, jsi::PropNameID::forAscii(rt, "run"), 1,
          [this](jsi::Runtime& runtime, const jsi::Value& thisVal,
                 const jsi::Value* args, size_t count) -> jsi::Value {
            return _model->runAsyncRaw(runtime, thisVal, args, count);
          });
    }

    if (name == "inputs") {
      return _model->getInputsRaw(rt, jsi::Value::undefined(), nullptr, 0);
    }

    if (name == "outputs") {
      return _model->getOutputsRaw(rt, jsi::Value::undefined(), nullptr, 0);
    }

    if (name == "delegate") {
      return _model->getDelegateRaw(rt, jsi::Value::undefined(), nullptr, 0);
    }

    if (name == "dispose") {
      return jsi::Function::createFromHostFunction(
          rt, jsi::PropNameID::forAscii(rt, "dispose"), 0,
          [](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
            return jsi::Value::undefined();
          });
    }

    return jsi::Value::undefined();
  }

  void set(jsi::Runtime&, const jsi::PropNameID&, const jsi::Value&) override {
    // Read-only object
  }

  std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& rt) override {
    std::vector<jsi::PropNameID> names;
    names.push_back(jsi::PropNameID::forAscii(rt, "runSync"));
    names.push_back(jsi::PropNameID::forAscii(rt, "run"));
    names.push_back(jsi::PropNameID::forAscii(rt, "inputs"));
    names.push_back(jsi::PropNameID::forAscii(rt, "outputs"));
    names.push_back(jsi::PropNameID::forAscii(rt, "delegate"));
    names.push_back(jsi::PropNameID::forAscii(rt, "dispose"));
    return names;
  }

private:
  std::shared_ptr<HybridTfliteModel> _model;
};
