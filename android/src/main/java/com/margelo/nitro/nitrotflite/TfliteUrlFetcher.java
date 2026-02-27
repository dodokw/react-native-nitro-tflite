package com.margelo.nitro.nitrotflite;

import android.annotation.SuppressLint;
import android.content.Context;
import android.net.Uri;
import android.util.Log;

import com.facebook.proguard.annotations.DoNotStrip;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.util.Objects;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

/**
 * Utility class for fetching byte data from URLs.
 * Called from C++ via JNI to load TFLite models.
 */
@DoNotStrip
public class TfliteUrlFetcher {
    private static final String TAG = "NitroTflite";
    private static WeakReference<Context> weakContext;
    private static final OkHttpClient client = new OkHttpClient();

    public static void setContext(Context context) {
        weakContext = new WeakReference<>(context);
    }

    @SuppressLint("DiscouragedApi")
    private static int getResourceId(Context context, String name) {
        return context.getResources().getIdentifier(
                name,
                "raw",
                context.getPackageName()
        );
    }

    @DoNotStrip
    public static byte[] fetchByteDataFromUrl(String url) throws Exception {
        Log.i(TAG, "Loading byte data from URL: " + url + "...");

        Uri uri = null;
        Integer resourceId = null;
        if (url.contains("://")) {
            uri = Uri.parse(url);
        } else {
            Context context = weakContext != null ? weakContext.get() : null;
            if (context != null) {
                resourceId = getResourceId(context, url);
            }
        }

        if (uri != null) {
            if (Objects.equals(uri.getScheme(), "file")) {
                // It's a file URL
                String path = Objects.requireNonNull(uri.getPath(), "File path cannot be null");
                File file = new File(path);

                if (!file.exists() || !file.canRead()) {
                    throw new IOException("File does not exist or is not readable: " + path);
                }

                if (!file.getName().toLowerCase().endsWith(".tflite")) {
                    throw new SecurityException("Only .tflite files are allowed");
                }

                try (FileInputStream stream = new FileInputStream(file)) {
                    return getLocalFileBytes(stream, file);
                }
            } else {
                // It's a network URL
                Request request = new Request.Builder().url(uri.toString()).build();
                try (Response response = client.newCall(request).execute()) {
                    if (response.isSuccessful() && response.body() != null) {
                        return response.body().bytes();
                    } else {
                        throw new RuntimeException("Response was not successful!");
                    }
                }
            }
        } else if (resourceId != null && resourceId != 0) {
            // It's bundled into the Android resources
            Context context = weakContext.get();
            if (context == null) {
                throw new Exception("React Context has already been destroyed!");
            }
            try (InputStream stream = context.getResources().openRawResource(resourceId)) {
                ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
                byte[] buffer = new byte[2048];
                int length;
                while ((length = stream.read(buffer)) != -1) {
                    byteStream.write(buffer, 0, length);
                }
                return byteStream.toByteArray();
            }
        } else {
            throw new Exception("Input is neither a valid URL, nor a resourceId - " +
                    "cannot load TFLite model! (Input: " + url + ")");
        }
    }

    private static byte[] getLocalFileBytes(InputStream stream, File file) throws IOException {
        long fileSize = file.length();
        if (fileSize > Integer.MAX_VALUE) {
            throw new IOException("File is too large to read into memory");
        }

        byte[] data = new byte[(int) fileSize];
        int bytesRead = 0;
        int chunk;
        while (bytesRead < fileSize && (chunk = stream.read(data, bytesRead, (int) fileSize - bytesRead)) != -1) {
            bytesRead += chunk;
        }

        if (bytesRead != fileSize) {
            throw new IOException("Could not completely read file " + file.getName());
        }

        return data;
    }
}
