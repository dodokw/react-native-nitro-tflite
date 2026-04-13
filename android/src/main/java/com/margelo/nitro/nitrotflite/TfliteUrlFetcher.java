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
import okhttp3.ResponseBody;

/**
 * Utility class for fetching byte data from URLs.
 * Called from C++ via JNI to load TFLite models.
 */
@DoNotStrip
public class TfliteUrlFetcher {
    private static final String TAG = "NitroTflite";
    private static WeakReference<Context> weakContext;
    private static final OkHttpClient client = new OkHttpClient();

    /** Callback interface for download progress. Called from JNI. */
    @DoNotStrip
    public interface ProgressListener {
        /** @param progress Value in [0.0, 1.0], or -1.0 if content-length is unknown. */
        void onProgress(double progress);
    }

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
        return fetchByteDataFromUrl(url, null);
    }

    @DoNotStrip
    public static byte[] fetchByteDataFromUrl(String url, ProgressListener listener) throws Exception {
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
                // Local file — no meaningful progress, report 100% at end.
                String path = Objects.requireNonNull(uri.getPath(), "File path cannot be null");
                File file = new File(path);

                if (!file.exists() || !file.canRead()) {
                    throw new IOException("File does not exist or is not readable: " + path);
                }

                if (!file.getName().toLowerCase().endsWith(".tflite")) {
                    throw new SecurityException("Only .tflite files are allowed");
                }

                try (FileInputStream stream = new FileInputStream(file)) {
                    byte[] data = getLocalFileBytes(stream, file);
                    if (listener != null) listener.onProgress(1.0);
                    return data;
                }
            } else {
                // Network URL — stream with progress.
                Request request = new Request.Builder().url(uri.toString()).build();
                try (Response response = client.newCall(request).execute()) {
                    if (!response.isSuccessful() || response.body() == null) {
                        throw new RuntimeException("Response was not successful: " + response.code());
                    }
                    return streamWithProgress(response.body(), listener);
                }
            }
        } else if (resourceId != null && resourceId != 0) {
            Context context = weakContext.get();
            if (context == null) {
                throw new Exception("React Context has already been destroyed!");
            }
            try (InputStream stream = context.getResources().openRawResource(resourceId)) {
                ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
                byte[] buffer = new byte[4096];
                int length;
                while ((length = stream.read(buffer)) != -1) {
                    byteStream.write(buffer, 0, length);
                }
                if (listener != null) listener.onProgress(1.0);
                return byteStream.toByteArray();
            }
        } else {
            throw new Exception("Input is neither a valid URL, nor a resourceId - " +
                    "cannot load TFLite model! (Input: " + url + ")");
        }
    }

    /** Stream a ResponseBody while reporting download progress. */
    private static byte[] streamWithProgress(ResponseBody body, ProgressListener listener)
            throws IOException {
        long contentLength = body.contentLength(); // -1 if unknown
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        long downloaded = 0;
        int read;
        try (InputStream in = body.byteStream()) {
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                downloaded += read;
                if (listener != null) {
                    double progress = contentLength > 0
                            ? (double) downloaded / (double) contentLength
                            : -1.0;
                    listener.onProgress(Math.min(progress, 1.0));
                }
            }
        }
        if (listener != null) listener.onProgress(1.0);
        return out.toByteArray();
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
