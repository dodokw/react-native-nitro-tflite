package com.margelo.nitro.nitrotflite

import com.facebook.proguard.annotations.DoNotStrip

/**
 * Loads the native C++ library for NitroTflite.
 * This is called from NitroTflitePackage companion init block.
 */
@DoNotStrip
object NitroTfliteOnLoad {
    @JvmStatic
    @DoNotStrip
    fun initializeNative() {
        try {
            System.loadLibrary("NitroTflite")
        } catch (e: UnsatisfiedLinkError) {
            throw RuntimeException("Failed to load NitroTflite native library!", e)
        }
    }
}
