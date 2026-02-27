project(NitroTflite)
cmake_minimum_required(VERSION 3.9.0)

set (PACKAGE_NAME NitroTflite)
set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_STANDARD 20)

string(APPEND CMAKE_CXX_FLAGS " -DANDROID")

# Find TFLite libraries
find_library(
  TFLITE
  tensorflowlite_jni
  PATHS "./src/main/cpp/lib/litert/jni/${ANDROID_ABI}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

find_library(
  TFLITE_GPU
  tensorflowlite_gpu_jni
  PATHS "./src/main/cpp/lib/litert/jni/${ANDROID_ABI}"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

# Define C++ library and add all sources
add_library(${PACKAGE_NAME} SHARED
  src/main/cpp/cpp-adapter.cpp
  ../cpp/HybridTfliteModel.cpp
  ../cpp/HybridTfliteModelFactory.cpp
  ../cpp/TensorHelpers.cpp
  ../cpp/jsi/Promise.cpp
  ../cpp/jsi/TypedArray.cpp
)

# Find NitroModules from node_modules
# When built as part of an app, our module is at: app/node_modules/react-native-nitro-tflite/android/
# So react-native-nitro-modules is at: app/node_modules/react-native-nitro-modules/
set(NITRO_MODULES_DIR "${CMAKE_SOURCE_DIR}/../node_modules/react-native-nitro-modules")
if(NOT EXISTS "${NITRO_MODULES_DIR}/cpp")
  # App's node_modules (hoisted): android/ -> ../ -> node_modules/react-native-nitro-tflite -> ../../ -> node_modules/
  set(NITRO_MODULES_DIR "${CMAKE_SOURCE_DIR}/../../react-native-nitro-modules")
endif()
if(NOT EXISTS "${NITRO_MODULES_DIR}/cpp")
  message(FATAL_ERROR "Could not find react-native-nitro-modules! Searched: ${CMAKE_SOURCE_DIR}/../node_modules and ${CMAKE_SOURCE_DIR}/../..")
endif()

# Set up includes
target_include_directories(
  ${PACKAGE_NAME}
  PRIVATE
  "src/main/cpp"
  "../cpp"
  "src/main/cpp/lib/litert/headers"
  "src/main/cpp/lib/litert/headers/tflite"
  # NitroModules headers are flattened to build/headers/NitroModules/ by gradle
  # Include the parent so #include <NitroModules/X.hpp> works
  "${CMAKE_SOURCE_DIR}/build/headers"
)

find_library(LOG_LIB log)

# Find React Native packages
find_package(ReactAndroid REQUIRED CONFIG)
find_package(fbjni REQUIRED CONFIG)

# Link base libraries
target_link_libraries(
  ${PACKAGE_NAME}
  ${LOG_LIB}
  android
  ReactAndroid::jsi
  fbjni::fbjni
  ${TFLITE}
  ${TFLITE_GPU}
)

# Find NitroModules .so - try prefab first, then use link directories
find_package(NitroModules CONFIG QUIET)

if(NitroModules_FOUND)
  message(STATUS "Found NitroModules via prefab")
  target_link_libraries(${PACKAGE_NAME} NitroModules::NitroModules)
else()
  # NitroModules is built by react-native-nitro-modules as a separate Gradle module.
  # At CMake configure time, the .so may not exist yet. Instead of find_library
  # (which checks at configure time), use target_link_directories to add the
  # search paths that the linker will use at link time.
  # We need to check BOTH nested (inside our module) and hoisted (app-level) paths
  # because npm/yarn may hoist react-native-nitro-modules to the app's node_modules.
  message(STATUS "NitroModules not found via prefab, using link directories for link-time resolution")
  target_link_directories(${PACKAGE_NAME} PRIVATE
    # Hoisted: app-level node_modules (most common)
    # CMAKE_SOURCE_DIR = .../node_modules/react-native-nitro-tflite/android/
    # ../../ = .../node_modules/
    "${CMAKE_SOURCE_DIR}/../../react-native-nitro-modules/android/build/intermediates/cmake/debug/obj/${ANDROID_ABI}"
    "${CMAKE_SOURCE_DIR}/../../react-native-nitro-modules/android/build/intermediates/cmake/release/obj/${ANDROID_ABI}"
    # From NITRO_MODULES_BUILD_DIR (Gradle-provided path)
    "${NITRO_MODULES_BUILD_DIR}/intermediates/cmake/debug/obj/${ANDROID_ABI}"
    "${NITRO_MODULES_BUILD_DIR}/intermediates/cmake/release/obj/${ANDROID_ABI}"
    # From NITRO_MODULES_DIR (CMake auto-detected, may be nested)
    "${NITRO_MODULES_DIR}/android/build/intermediates/cmake/debug/obj/${ANDROID_ABI}"
    "${NITRO_MODULES_DIR}/android/build/intermediates/cmake/release/obj/${ANDROID_ABI}"
  )
  target_link_libraries(${PACKAGE_NAME} NitroModules)
endif()

if(ReactAndroid_VERSION_MINOR GREATER_EQUAL 76)
  target_link_libraries(
    ${PACKAGE_NAME}
    ReactAndroid::reactnative
  )
else()
  target_link_libraries(
    ${PACKAGE_NAME}
    ReactAndroid::reactnativejni
  )
endif()

target_link_options(${PACKAGE_NAME}
  PRIVATE
  "-Wl,-z,max-page-size=16384"
  "-Wl,-z,common-page-size=16384"
)
