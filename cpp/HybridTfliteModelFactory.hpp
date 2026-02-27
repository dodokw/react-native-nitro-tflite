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
#include <string>

using namespace facebook;
using namespace margelo::nitro;

typedef std::function<Buffer(std::string)> FetchURLFunc;

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

protected:
  void loadHybridMethods() override;
  size_t getExternalMemorySize() noexcept override;

private:
  // Raw JSI method: loadModel(path: string, delegate: string) => Promise<TfliteModel>
  jsi::Value loadModelRaw(jsi::Runtime& runtime, const jsi::Value& thisValue,
                          const jsi::Value* args, size_t count);

  static HybridTfliteModel::Delegate parseDelegateString(const std::string& delegate);

private:
  FetchURLFunc _fetchURL;
  static std::shared_ptr<HybridTfliteModelFactory> _instance;

  static constexpr auto TAG = "TfliteModelFactory";
};
