#include <jni.h>
#include <fbjni/fbjni.h>
#include <NitroModules/HybridObjectRegistry.hpp>
#include "HybridTfliteModelFactory.hpp"
#include <android/log.h>
#include <string>

#define LOG_TAG "NitroTflite"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace nitrotflite {

JavaVM* java_machine = nullptr;

// Cached JNI references (must be global refs to survive across threads)
static jclass gTfliteUrlFetcherClass = nullptr;
static jmethodID gFetchByteDataMethod = nullptr;

using namespace facebook;
using namespace facebook::jni;
using namespace margelo::nitro;

struct NitroTfliteOnLoad : public jni::JavaClass<NitroTfliteOnLoad> {
public:
  static constexpr auto kJavaDescriptor = "Lcom/margelo/nitro/nitrotflite/NitroTfliteOnLoad;";

  static void registerNatives() {
    // No JNI methods needed - we use HybridObjectRegistry directly
  }
};

/**
 * Extract the message from a Java exception via JNI.
 */
static std::string getJavaExceptionMessage(JNIEnv* env, jthrowable exception) {
  jclass throwableClass = env->FindClass("java/lang/Throwable");
  jmethodID getMessageMethod = env->GetMethodID(throwableClass, "getMessage", "()Ljava/lang/String;");
  auto jMsg = (jstring)env->CallObjectMethod(exception, getMessageMethod);

  if (jMsg != nullptr) {
    const char* chars = env->GetStringUTFChars(jMsg, nullptr);
    std::string message(chars);
    env->ReleaseStringUTFChars(jMsg, chars);
    env->DeleteLocalRef(jMsg);
    env->DeleteLocalRef(throwableClass);
    return message;
  }

  // Fallback: try toString()
  jmethodID toStringMethod = env->GetMethodID(throwableClass, "toString", "()Ljava/lang/String;");
  auto jStr = (jstring)env->CallObjectMethod(exception, toStringMethod);
  env->DeleteLocalRef(throwableClass);

  if (jStr != nullptr) {
    const char* chars = env->GetStringUTFChars(jStr, nullptr);
    std::string message(chars);
    env->ReleaseStringUTFChars(jStr, chars);
    env->DeleteLocalRef(jStr);
    return message;
  }

  return "Unknown Java exception";
}

} // namespace nitrotflite

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  nitrotflite::java_machine = vm;

  return facebook::jni::initialize(vm, [vm] {
    // Cache JNI class and method references on the main thread
    // (FindClass only works reliably from the main thread's class loader)
    JNIEnv* env = nullptr;
    vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (env != nullptr) {
      jclass localClass = env->FindClass("com/margelo/nitro/nitrotflite/TfliteUrlFetcher");
      if (localClass != nullptr) {
        nitrotflite::gTfliteUrlFetcherClass = (jclass)env->NewGlobalRef(localClass);
        nitrotflite::gFetchByteDataMethod = env->GetStaticMethodID(
          nitrotflite::gTfliteUrlFetcherClass, "fetchByteDataFromUrl", "(Ljava/lang/String;)[B");
        env->DeleteLocalRef(localClass);
        LOGI("Cached TfliteUrlFetcher class and method references");
      } else {
        LOGE("Failed to find TfliteUrlFetcher class during JNI_OnLoad!");
        env->ExceptionClear();
      }
    }

    // Register the TfliteModelFactory in the HybridObjectRegistry
    margelo::nitro::HybridObjectRegistry::registerHybridObjectConstructor(
      "TfliteModelFactory",
      [vm]() -> std::shared_ptr<margelo::nitro::HybridObject> {
        auto factory = HybridTfliteModelFactory::getOrCreate();

        // Set up the Android-specific URL fetcher using JNI
        factory->setFetchURLFunc([vm](std::string url) -> Buffer {
          JNIEnv* env = nullptr;
          int getEnvStat = vm->GetEnv((void**)&env, JNI_VERSION_1_6);
          bool didAttach = false;
          if (getEnvStat == JNI_EDETACHED) {
            if (vm->AttachCurrentThread(&env, nullptr) != 0) {
              throw std::runtime_error("Failed to attach thread to JVM");
            }
            didAttach = true;
          }

          LOGI("Fetching model from URL: %s", url.c_str());

          // Use cached class and method references (safe from any thread)
          if (nitrotflite::gTfliteUrlFetcherClass == nullptr || nitrotflite::gFetchByteDataMethod == nullptr) {
            if (didAttach) vm->DetachCurrentThread();
            throw std::runtime_error("TfliteUrlFetcher JNI references not initialized!");
          }

          jstring jUrl = env->NewStringUTF(url.c_str());
          auto byteArray = (jbyteArray)env->CallStaticObjectMethod(
            nitrotflite::gTfliteUrlFetcherClass, nitrotflite::gFetchByteDataMethod, jUrl);

          // Check for Java exception
          if (env->ExceptionCheck()) {
            jthrowable exc = env->ExceptionOccurred();
            env->ExceptionClear();
            std::string message = nitrotflite::getJavaExceptionMessage(env, exc);
            LOGE("Java exception during fetch: %s", message.c_str());
            env->DeleteLocalRef(exc);
            env->DeleteLocalRef(jUrl);
            if (didAttach) vm->DetachCurrentThread();
            throw std::runtime_error("Failed to fetch model: " + message);
          }

          if (byteArray == nullptr) {
            env->DeleteLocalRef(jUrl);
            if (didAttach) vm->DetachCurrentThread();
            throw std::runtime_error("fetchByteDataFromUrl returned null for URL: " + url);
          }

          jsize size = env->GetArrayLength(byteArray);
          void* data = malloc(size);
          env->GetByteArrayRegion(byteArray, 0, size, (jbyte*)data);

          LOGI("Fetched %d bytes successfully", (int)size);

          env->DeleteLocalRef(byteArray);
          env->DeleteLocalRef(jUrl);
          if (didAttach) vm->DetachCurrentThread();

          return Buffer{.data = data, .size = static_cast<size_t>(size)};
        });

        return factory;
      }
    );
  });
}
