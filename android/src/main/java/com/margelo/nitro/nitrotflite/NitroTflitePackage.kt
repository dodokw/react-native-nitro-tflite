package com.margelo.nitro.nitrotflite

import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfoProvider
import com.facebook.react.BaseReactPackage

class NitroTflitePackage : BaseReactPackage() {
    override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? {
        // Set the context for the URL fetcher
        TfliteUrlFetcher.setContext(reactContext)
        return null
    }

    override fun getReactModuleInfoProvider(): ReactModuleInfoProvider = ReactModuleInfoProvider { HashMap() }

    companion object {
        init {
            // Load NitroModules first since NitroTflite depends on it
            try {
                System.loadLibrary("NitroModules")
            } catch (_: UnsatisfiedLinkError) {
                // NitroModules may already be loaded by react-native-nitro-modules
            }
            NitroTfliteOnLoad.initializeNative()
        }
    }
}
