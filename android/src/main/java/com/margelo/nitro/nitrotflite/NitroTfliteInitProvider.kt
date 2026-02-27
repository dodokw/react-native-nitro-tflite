package com.margelo.nitro.nitrotflite

import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.net.Uri
import android.util.Log

/**
 * Auto-initializer that sets up the TfliteUrlFetcher context.
 * ContentProvider is instantiated before Application.onCreate(),
 * so the context is available when the native library needs it.
 */
class NitroTfliteInitProvider : ContentProvider() {
    companion object {
        private const val TAG = "NitroTflite"
    }

    override fun onCreate(): Boolean {
        val ctx = context
        if (ctx != null) {
            Log.i(TAG, "NitroTfliteInitProvider: setting context for TfliteUrlFetcher")
            TfliteUrlFetcher.setContext(ctx.applicationContext)
        } else {
            Log.w(TAG, "NitroTfliteInitProvider: context is null!")
        }
        return true
    }

    override fun query(uri: Uri, projection: Array<String>?, selection: String?, selectionArgs: Array<String>?, sortOrder: String?): Cursor? = null
    override fun getType(uri: Uri): String? = null
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<String>?): Int = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<String>?): Int = 0
}
