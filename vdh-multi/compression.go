package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/klauspost/compress/zstd"
)

type zstdReadCloser struct {
	reader *zstd.Decoder
	closer io.Closer
}

func (z *zstdReadCloser) Read(p []byte) (n int, err error) {
	return z.reader.Read(p)
}

func (z *zstdReadCloser) Close() error {
	z.reader.Close()
	return z.closer.Close()
}

type zstdResponseWriter struct {
	http.ResponseWriter
	writer *zstd.Encoder
}

func (w *zstdResponseWriter) Write(b []byte) (int, error) {
	return w.writer.Write(b)
}

func (w *zstdResponseWriter) Close() error {
	return w.writer.Close()
}

type flushingResponseWriter struct {
	http.ResponseWriter
	flusher http.Flusher
}

func (w *flushingResponseWriter) Write(b []byte) (int, error) {
	n, err := w.ResponseWriter.Write(b)
	if w.flusher != nil {
		w.flusher.Flush()
	}
	return n, err
}

func decompressRequest(r *http.Request) error {
	if !strings.EqualFold(r.Header.Get("Content-Encoding"), "zstd") {
		return nil
	}

	reader, err := zstd.NewReader(r.Body)
	if err != nil {
		return fmt.Errorf("zstd reader creation failed: %v", err)
	}

	r.Body = &zstdReadCloser{
		reader: reader,
		closer: r.Body,
	}

	// Set ContentLength to -1 to indicate unknown length after decompression
	r.ContentLength = -1
	r.Header.Del("Content-Length")
	r.Header.Del("Content-Encoding")

	return nil
}

func compressionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		acceptEncoding := r.Header.Get("Accept-Encoding")

		if strings.Contains(acceptEncoding, "zstd") {
			encoder, err := zstd.NewWriter(w)
			if err != nil {
				log.Printf("Failed to create zstd encoder: %v", err)
				next.ServeHTTP(w, r)
				return
			}

			w.Header().Set("Content-Encoding", "zstd")
			w.Header().Del("Content-Length")

			zw := &zstdResponseWriter{
				ResponseWriter: w,
				writer:         encoder,
			}
			defer zw.Close()

			next.ServeHTTP(zw, r)
		} else {
			// For non-compressed responses, use flushing writer for immediate streaming
			if flusher, ok := w.(http.Flusher); ok {
				fw := &flushingResponseWriter{
					ResponseWriter: w,
					flusher:        flusher,
				}
				next.ServeHTTP(fw, r)
			} else {
				next.ServeHTTP(w, r)
			}
		}
	})
}
