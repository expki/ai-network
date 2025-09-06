package main

import (
	"crypto/tls"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
)

func main() {
	listenAddr := getEnvOrDefault("LISTEN_ADDR", ":5000")
	targets, err := parseTargetURLs()
	if err != nil {
		log.Fatal(err)
	}

	cert, err := generateCertificate()
	if err != nil {
		log.Fatalf("Failed to generate certificate: %v", err)
	}
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		NextProtos:   []string{"h2", "http/1.1"},
		ClientAuth:   tls.NoClientCert,
	}

	proxy := createReverseProxy(targets.chat, targets.embed, targets.rerank)
	handler := authMiddleware(compressionMiddleware(proxy))

	server := &http.Server{
		Addr:      listenAddr,
		Handler:   handler,
		TLSConfig: tlsConfig,
	}
	log.Printf("Starting HTTPS reverse proxy on %s", listenAddr)
	log.Printf("Chat target: %s", targets.chat.String())
	log.Printf("Embed target: %s", targets.embed.String())
	log.Printf("Rerank target: %s", targets.rerank.String())
	if bearerToken != "" {
		log.Println("Bearer token authentication enabled")
	} else {
		log.Println("Bearer token authentication disabled (allowing all requests)")
	}

	if err := server.ListenAndServeTLS("", ""); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

type targetURLs struct {
	chat   *url.URL
	embed  *url.URL
	rerank *url.URL
}

func parseTargetURLs() (targetURLs, error) {
	// Get URLs from environment variables with defaults
	chatURL := getEnvOrDefault("TARGET_URL_CHAT", "http://localhost:5001")
	embedURL := getEnvOrDefault("TARGET_URL_EMBED", "http://localhost:5002")
	rerankURL := getEnvOrDefault("TARGET_URL_RERANK", "http://localhost:5003")

	chat, err := url.Parse(chatURL)
	if err != nil {
		return targetURLs{}, fmt.Errorf("parsing chat %q: %v", rerankURL, err)
	}
	chat.Path = ""

	embed, err := url.Parse(embedURL)
	if err != nil {
		return targetURLs{}, fmt.Errorf("parsing embed %q: %v", rerankURL, err)
	}
	embed.Path = ""

	rerank, err := url.Parse(rerankURL)
	if err != nil {
		return targetURLs{}, fmt.Errorf("parsing rerank %q: %v", rerankURL, err)
	}
	rerank.Path = ""

	return targetURLs{
		chat:   chat,
		embed:  embed,
		rerank: rerank,
	}, nil
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
