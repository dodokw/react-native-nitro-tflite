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
#include <mutex>
#include <string>
#include <unordered_map>

using namespace facebook;

class TfliteModelHostObject : public jsi::HostObject {
public:
  explicit TfliteModelHostObject(std::shared_ptr<HybridTfliteModel> model)
      : _model(std::move(model)) {}

  jsi::Value get(jsi::Runtime& rt, const jsi::PropNameID& propName) override {
    auto name = propName.utf8(rt);

    if (name == "runSync") {
      return getCachedFunction(rt, "runSync", 1,
          [this](jsi::Runtime& runtime, const jsi::Value& thisVal,
                 const jsi::Value* args, size_t count) -> jsi::Value {
            return _model->runSyncRaw(runtime, thisVal, args, count);
          });
    }

    if (name == "run") {
      return getCachedFunction(rt, "run", 1,
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
      return getCachedFunction(rt, "dispose", 0,
          [](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
            return jsi::Value::undefined();
          });
    }

    if (name == "reshapeInput") {
      return getCachedFunction(rt, "reshapeInput", 2,
          [this](jsi::Runtime& runtime, const jsi::Value& thisVal,
                 const jsi::Value* args, size_t count) -> jsi::Value {
            return _model->reshapeInputRaw(runtime, thisVal, args, count);
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
    names.push_back(jsi::PropNameID::forAscii(rt, "reshapeInput"));
    return names;
  }

private:
  std::shared_ptr<HybridTfliteModel> _model;

  // ── Function cache per runtime ──────────────────────────────────────────
  // jsi::Function is bound to a specific jsi::Runtime. When the same
  // HostObject is accessed from multiple runtimes (e.g., JS thread +
  // Worklet runtime), each runtime needs its own cached copy.
  //
  // Key: (runtime pointer, function name) → cached jsi::Function
  // Using a mutex because Frame Processor threads may call concurrently.
  struct RuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<jsi::Function>> functions;
  };
  std::unordered_map<uintptr_t, RuntimeCache> _cachePerRuntime;
  std::mutex _cacheMutex;

  /**
   * Return a cached jsi::Function for the given runtime+name, creating it
   * on first access. This avoids allocating a new jsi::Function + lambda
   * on every property access (which was the main cause of memory growth).
   */
  template <typename HostFn>
  jsi::Value getCachedFunction(jsi::Runtime& rt, const char* name,
                               unsigned int paramCount, HostFn&& fn) {
    auto runtimeKey = reinterpret_cast<uintptr_t>(&rt);
    std::lock_guard<std::mutex> lock(_cacheMutex);

    auto& cache = _cachePerRuntime[runtimeKey];
    auto it = cache.functions.find(name);
    if (it != cache.functions.end()) {
      return jsi::Value(rt, *it->second);
    }

    auto func = std::make_shared<jsi::Function>(
        jsi::Function::createFromHostFunction(
            rt, jsi::PropNameID::forAscii(rt, name), paramCount,
            std::forward<HostFn>(fn)));
    cache.functions[name] = func;
    return jsi::Value(rt, *func);
  }
};
