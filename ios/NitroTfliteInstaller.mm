//
//  NitroTfliteInstaller.mm
//  react-native-nitro-tflite
//
//  Registers the TfliteModelFactory HybridObject in the Nitro HybridObjectRegistry.
//  Sets up the platform-specific URL fetcher for iOS (with progress reporting).
//

#import <Foundation/Foundation.h>
#import <NitroModules/HybridObjectRegistry.hpp>
#import "HybridTfliteModelFactory.hpp"

// ── ProgressDelegate ──────────────────────────────────────────────────────────
// NSURLSessionDataDelegate that streams the response body while reporting
// download progress via the C++ ProgressCallback.
// -----------------------------------------------------------------
@interface _TfliteProgressDelegate : NSObject <NSURLSessionDataDelegate>
@property (nonatomic, assign) long long expectedLength;
@property (nonatomic, assign) long long receivedLength;
@property (nonatomic, copy)   void (^progressHandler)(double);
@property (nonatomic, copy)   void (^completionHandler)(NSData * _Nullable, NSError * _Nullable);
@property (nonatomic, strong) NSMutableData* buffer;
@end

@implementation _TfliteProgressDelegate
- (instancetype)initWithProgress:(void(^)(double))progressHandler
                      completion:(void(^)(NSData*, NSError*))completionHandler {
  if ((self = [super init])) {
    _progressHandler = [progressHandler copy];
    _completionHandler = [completionHandler copy];
    _buffer = [NSMutableData data];
    _expectedLength = NSURLResponseUnknownLength;
  }
  return self;
}
- (void)URLSession:(NSURLSession *)session
          dataTask:(NSURLSessionDataTask *)dataTask
didReceiveResponse:(NSURLResponse *)response
 completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler {
  _expectedLength = response.expectedContentLength;
  completionHandler(NSURLSessionResponseAllow);
}
- (void)URLSession:(NSURLSession *)session
          dataTask:(NSURLSessionDataTask *)dataTask
    didReceiveData:(NSData *)data {
  [_buffer appendData:data];
  _receivedLength += (long long)data.length;
  if (_progressHandler && _expectedLength > 0) {
    double progress = (double)_receivedLength / (double)_expectedLength;
    _progressHandler(MIN(progress, 1.0));
  } else if (_progressHandler) {
    // Unknown content-length: report indeterminate progress.
    _progressHandler(-1.0);
  }
}
- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
didCompleteWithError:(NSError *)error {
  if (error) {
    _completionHandler(nil, error);
  } else {
    _completionHandler([_buffer copy], nil);
  }
}
@end

// ── NitroTfliteInstaller ──────────────────────────────────────────────────────
@interface NitroTfliteInstaller : NSObject
@end

@implementation NitroTfliteInstaller

+ (void)load {
  margelo::nitro::HybridObjectRegistry::registerHybridObjectConstructor(
    "TfliteModelFactory",
    []() -> std::shared_ptr<margelo::nitro::HybridObject> {
      auto factory = HybridTfliteModelFactory::getOrCreate();

      factory->setFetchURLFunc([](std::string url, ProgressCallback onProgress) -> Buffer {
        NSString* urlString = [NSString stringWithUTF8String:url.c_str()];
        NSURL* nsURL = [NSURL URLWithString:urlString];

        // ── Local file:// path — read directly, no progress needed ─────────
        if ([nsURL.scheme isEqualToString:@"file"]) {
          NSData* contents = [NSData dataWithContentsOfURL:nsURL];
          if (contents == nil) {
            throw std::runtime_error("Failed to read file: " + url);
          }
          if (onProgress) onProgress(1.0);
          void* data = malloc(contents.length);
          memcpy(data, contents.bytes, contents.length);
          return Buffer{.data = data, .size = contents.length};
        }

        // ── HTTP(S) — stream with progress tracking ────────────────────────
        __block NSData* resultData = nil;
        __block NSError* resultError = nil;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);

        void (^progressHandler)(double) = nil;
        if (onProgress) {
          progressHandler = ^(double p){ onProgress(p); };
        }

        _TfliteProgressDelegate* delegate =
            [[_TfliteProgressDelegate alloc]
                initWithProgress:progressHandler
                      completion:^(NSData* data, NSError* err) {
                        resultData = data;
                        resultError = err;
                        dispatch_semaphore_signal(sem);
                      }];

        NSURLSession* session = [NSURLSession sessionWithConfiguration:
            [NSURLSessionConfiguration defaultSessionConfiguration]
            delegate:delegate delegateQueue:nil];

        [[session dataTaskWithURL:nsURL] resume];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        [session invalidateAndCancel];

        if (resultError != nil || resultData == nil) {
          std::string msg = resultError
              ? std::string(resultError.localizedDescription.UTF8String)
              : "No data received from URL";
          throw std::runtime_error("Failed to fetch model from " + url + ": " + msg);
        }

        void* data = malloc(resultData.length);
        memcpy(data, resultData.bytes, resultData.length);
        return Buffer{.data = data, .size = resultData.length};
      });

      return factory;
    }
  );
}

@end
