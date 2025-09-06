package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io"
	"log"
	"math/big"
	mrand "math/rand/v2"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"
)

var bearerToken string

func init() {
	bearerToken = os.Getenv("BEARER_TOKEN")
}

func generateCertificate() (tls.Certificate, error) {
	priv, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to generate private key: %v", err)
	}

	template := x509.Certificate{
		SerialNumber: big.NewInt(int64(mrand.Uint32())),
		Subject: pkix.Name{
			Organization:  []string{"LlamaCPP Proxy"},
			Province:      []string{""},
			Locality:      []string{""},
			StreetAddress: []string{""},
			PostalCode:    []string{""},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
		DNSNames:              []string{"localhost", "llama.cpp"},
	}

	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to create certificate: %v", err)
	}

	privKeyDER, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to marshal private key: %v", err)
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: privKeyDER})

	return tls.X509KeyPair(certPEM, keyPEM)
}

func authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if bearerToken != "" {
			auth := r.Header.Get("Authorization")
			if auth == "" {
				http.Error(w, "Unauthorized: Missing Authorization header", http.StatusUnauthorized)
				return
			}

			const prefix = "Bearer "
			if !strings.HasPrefix(auth, prefix) {
				http.Error(w, "Unauthorized: Invalid Authorization format", http.StatusUnauthorized)
				return
			}

			token := strings.TrimPrefix(auth, prefix)
			if token != bearerToken {
				http.Error(w, "Unauthorized: Invalid token", http.StatusUnauthorized)
				return
			}
		}

		next.ServeHTTP(w, r)
	})
}

func decompressRequest(r *http.Request) error {
	if strings.EqualFold(r.Header.Get("Content-Encoding"), "zstd") {
		reader, err := zstd.NewReader(r.Body)
		if err != nil {
			return fmt.Errorf("zstd reader creation failed: %v", err)
		}

		r.Body = &zstdReadCloser{
			reader: reader,
			closer: r.Body,
		}
		r.ContentLength = -1 // Unknown after decompression
		r.Header.Del("Content-Encoding")
	}
	return nil
}

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

			// Set Content-Encoding header before creating the response writer
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

func createReverseProxy(chatTarget, embedTarget, rerankTarget *url.URL) *httputil.ReverseProxy {
	chatPattern := regexp.MustCompile(`/(?:v1/)?(?:(?:chat/)?completions?|completion|complete|infill)/?$`)
	embedPattern := regexp.MustCompile(`/(?:v1/)?embeddings?/?$`)
	rerankPattern := regexp.MustCompile(`/(?:v1/)?rerank(?:ing)?/?$`)

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			var target *url.URL

			path := req.URL.Path
			queryType := req.Header.Get("Query-Type")
			if queryType != "" {
				req.Header.Del(queryType)
			}

			if strings.EqualFold(queryType, "chat") || (queryType == "" && chatPattern.MatchString(path)) {
				log.Println("Chat query detected")
				target = chatTarget
			} else if strings.EqualFold(queryType, "embed") || queryType == "" && embedPattern.MatchString(path) {
				log.Println("Embed query detected")
				target = embedTarget
			} else if strings.EqualFold(queryType, "rerank") || queryType == "" && rerankPattern.MatchString(path) {
				log.Println("Rerank query detected")
				target = rerankTarget
			} else {
				log.Println("No query detected, fallback to Chat")
				target = chatTarget
			}

			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.URL.Path = singleJoiningSlash(target.Path, req.URL.Path)
			if target.RawQuery == "" || req.URL.RawQuery == "" {
				req.URL.RawQuery = target.RawQuery + req.URL.RawQuery
			} else {
				req.URL.RawQuery = target.RawQuery + "&" + req.URL.RawQuery
			}

			if err := decompressRequest(req); err != nil {
				log.Printf("Failed to decompress request: %v", err)
			}

			acceptEncoding := req.Header.Get("Accept-Encoding")
			if strings.Contains(strings.ToLower(acceptEncoding), "zstd") {
				req.Header.Set("Accept-Encoding", "zstd")
			} else {
				req.Header.Del("Accept-Encoding")
			}

			req.Header.Del("Authorization")
			req.Host = target.Host
		},
		ModifyResponse: func(resp *http.Response) error {
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("Proxy error: %v", err)
			http.Error(w, "Bad Gateway", http.StatusBadGateway)
		},
	}

	return proxy
}

func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}

func main() {
	chatTargetURL := os.Getenv("TARGET_URL_CHAT")
	if chatTargetURL == "" {
		chatTargetURL = "http://localhost:5001"
	}
	embedTargetURL := os.Getenv("TARGET_URL_EMBED")
	if embedTargetURL == "" {
		embedTargetURL = "http://localhost:5002"
	}
	rerankTargetURL := os.Getenv("TARGET_URL_RERANK")
	if rerankTargetURL == "" {
		rerankTargetURL = "http://localhost:5003"
	}

	chatTarget, err := url.Parse(chatTargetURL)
	if err != nil {
		log.Fatalf("Invalid target Chat URL: %v", err)
	}
	embedTarget, err := url.Parse(embedTargetURL)
	if err != nil {
		log.Fatalf("Invalid target Embed URL: %v", err)
	}
	rerankTarget, err := url.Parse(rerankTargetURL)
	if err != nil {
		log.Fatalf("Invalid target Rerank URL: %v", err)
	}

	listenAddr := os.Getenv("LISTEN_ADDR")
	if listenAddr == "" {
		listenAddr = ":5000"
	}

	cert, err := generateCertificate()
	if err != nil {
		log.Fatalf("Failed to generate certificate: %v", err)
	}

	proxy := createReverseProxy(chatTarget, embedTarget, rerankTarget)

	handler := authMiddleware(compressionMiddleware(proxy))

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{"h2", "http/1.1"},
		ClientAuth:   tls.NoClientCert,
	}

	server := &http.Server{
		Addr:      listenAddr,
		Handler:   handler,
		TLSConfig: tlsConfig,
	}

	log.Printf("Starting HTTPS reverse proxy on %s", listenAddr)
	log.Printf("Chat target: %s", chatTarget.String())
	log.Printf("Embed target: %s", embedTarget.String())
	log.Printf("Rerank target: %s", rerankTarget.String())
	if bearerToken != "" {
		log.Println("Bearer token authentication enabled")
	} else {
		log.Println("Bearer token authentication disabled (allowing all requests)")
	}

	if err := server.ListenAndServeTLS("", ""); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
