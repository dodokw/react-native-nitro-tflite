require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

enableCoreMLDelegate = false
if defined?($EnableCoreMLDelegate)
  enableCoreMLDelegate = $EnableCoreMLDelegate
end

enableMetalDelegate = false
if defined?($EnableMetalDelegate)
  enableMetalDelegate = $EnableMetalDelegate
end

Pod::UI.puts "[NitroTflite] CoreML Delegate: #{enableCoreMLDelegate}  Metal Delegate: #{enableMetalDelegate}"

Pod::Spec.new do |s|
  s.name         = "NitroTflite"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "13.0" }
  s.source       = { :git => "https://github.com/dodokw/react-native-nitro-tflite.git", :tag => "#{s.version}" }

  s.source_files = [
    # Implementation (Objective-C++)
    "ios/**/*.{h,m,mm}",
    # Implementation (C++ objects)
    "cpp/**/*.{hpp,cpp,c,h}",
  ]

  # Add Nitrogen-generated files for autolinking
  load 'nitrogen/generated/ios/NitroTflite+autolinking.rb'
  add_nitrogen_files(s)

  preprocessor_flags = "$(inherited)"
  preprocessor_flags += " FAST_TFLITE_ENABLE_CORE_ML=#{enableCoreMLDelegate ? 1 : 0}"
  preprocessor_flags += " FAST_TFLITE_ENABLE_METAL=#{enableMetalDelegate ? 1 : 0}"

  s.pod_target_xcconfig = {
    'GCC_PREPROCESSOR_DEFINITIONS' => preprocessor_flags,
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++20',
  }

  # TensorFlow Lite C API
  s.dependency "TensorFlowLiteC", "2.17.0"

  if enableCoreMLDelegate
    s.dependency "TensorFlowLiteC/CoreML", "2.17.0"
  end

  if enableMetalDelegate
    s.dependency "TensorFlowLiteCMetal", "2.17.0"
  end

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  install_modules_dependencies(s)
end
