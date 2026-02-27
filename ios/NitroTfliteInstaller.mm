//
//  NitroTfliteInstaller.mm
//  react-native-nitro-tflite
//
//  Registers the TfliteModelFactory HybridObject in the Nitro HybridObjectRegistry.
//  Sets up the platform-specific URL fetcher for iOS.
//

#import <Foundation/Foundation.h>
#import <NitroModules/HybridObjectRegistry.hpp>
#import "HybridTfliteModelFactory.hpp"

@interface NitroTfliteInstaller : NSObject
@end

@implementation NitroTfliteInstaller

+ (void)load {
  // Register the TfliteModelFactory in the HybridObjectRegistry
  // This runs automatically when the library is loaded
  margelo::nitro::HybridObjectRegistry::registerHybridObjectConstructor(
    "TfliteModelFactory",
    []() -> std::shared_ptr<margelo::nitro::HybridObject> {
      auto factory = HybridTfliteModelFactory::getOrCreate();

      // Set up the iOS-specific URL fetcher
      factory->setFetchURLFunc([](std::string url) -> Buffer {
        NSString* string = [NSString stringWithUTF8String:url.c_str()];
        NSURL* nsURL = [NSURL URLWithString:string];
        NSData* contents = [NSData dataWithContentsOfURL:nsURL];

        if (contents == nil) {
          throw std::runtime_error("Failed to fetch data from URL: " + url);
        }

        void* data = malloc(contents.length * sizeof(uint8_t));
        memcpy(data, contents.bytes, contents.length);
        return Buffer{.data = data, .size = contents.length};
      });

      return factory;
    }
  );
}

@end
