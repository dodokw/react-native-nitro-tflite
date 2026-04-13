//
//  HybridTfliteModelFactory.hpp
//  react-native-nitro-tflite
//
//  Factory HybridObject for loading TFLite models.
//  Registered in the HybridObjectRegistry so it can be created from JS.
//

#pragma once

#include <NitroModules/HybridObject.hpp>
#include "HybridTfliteModel.hpp"
#include <jsi/jsi.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace facebook;
using namespace margelo::nitro;

// ProgressCallback: called with progress in [0.0, 1.0]. -1.0 means unknown.
typedef std::function<void(double)> ProgressCallback;

// FetchURLFunc: fetches model bytes from a URL.
// - url:      The model source URI
// - onProgress: optional progress callback (may be nullptr)
typedef std::function<Buffer(std::string, ProgressCallback)> FetchURLFunc;

class HybridTfliteModelFactory : public HybridObject {
public:
  explicit HybridTfliteModelFactory();

  /**
   * Set the platform-specific URL fetcher.
   * This must be called by native platform code (iOS/Android) before any model loading.
   */
  void setFetchURLFunc(FetchURLFunc fetchFunc);

  /**
   * Get the singleton factory instance.
   */
  static std::shared_ptr<HybridTfliteModelFactory> getOrCreate();

  /**
   * Clear the in-memory model cache. Expired (GC'd) entries are always
   * cleaned up automatically; call this to eagerly release all cached models.
   */
  void clearModelCache();

protected:
  void loadHybridMethods() override;
  size_t getExternalMemorySize() noexcept override;

private:
  // Raw JSI method: loadModel(path, delegate, onProgress?) => Promise<TfliteModel>
  jsi::Value loadModelRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                          const jsi::Value* args, size_t count);
  // Raw JSI method: clearCache() => void
  jsi::Value clearCacheRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                           const jsi::Value* args, size_t count);

  static HybridTfliteModel::Delegate parseDelegateString(const std::string& delegate);

private:
  FetchURLFunc _fetchURL;

  // Cache key = "<url>:<delegate>"  →  weak_ptr so the model can still be GC'd
  std::unordered_map<std::string, std::weak_ptr<HybridTfliteModel>> _modelCache;
  std::mutex _cacheMutex;

  static std::shared_ptr<HybridTfliteModelFactory> _instance;
  static std::once_flag _instanceFlag;

  static constexpr auto TAG = "TfliteModelFactory";
};
