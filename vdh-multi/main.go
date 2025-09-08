package main

import (
	"crypto/tls"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
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
	log.Printf("Chat targets (%d): %s", len(targets.chat), formatURLs(targets.chat))
	log.Printf("Embed targets (%d): %s", len(targets.embed), formatURLs(targets.embed))
	log.Printf("Rerank targets (%d): %s", len(targets.rerank), formatURLs(targets.rerank))
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
	chat   []*url.URL
	embed  []*url.URL
	rerank []*url.URL
}

func parseTargetURLs() (targetURLs, error) {
	// Get URLs from environment variables with defaults
	chatURLs := getEnvOrDefault("TARGET_URL_CHAT", "http://localhost:6000")
	embedURLs := getEnvOrDefault("TARGET_URL_EMBED", "http://localhost:7000")
	rerankURLs := getEnvOrDefault("TARGET_URL_RERANK", "http://localhost:8000")

	parsedChat, err := parseURLList(chatURLs, "chat")
	if err != nil {
		return targetURLs{}, err
	}

	parsedEmbed, err := parseURLList(embedURLs, "embed")
	if err != nil {
		return targetURLs{}, err
	}

	parsedRerank, err := parseURLList(rerankURLs, "rerank")
	if err != nil {
		return targetURLs{}, err
	}

	return targetURLs{
		chat:   parsedChat,
		embed:  parsedEmbed,
		rerank: parsedRerank,
	}, nil
}

func parseURLList(urlList string, name string) ([]*url.URL, error) {
	urlStrings := strings.Split(urlList, ",")
	var urls []*url.URL

	for _, urlStr := range urlStrings {
		urlStr = strings.TrimSpace(urlStr)
		if urlStr == "" {
			continue
		}
		parsed, err := url.Parse(urlStr)
		if err != nil {
			return nil, fmt.Errorf("parsing %s URL %q: %v", name, urlStr, err)
		}
		parsed.Path = ""
		urls = append(urls, parsed)
	}

	if len(urls) == 0 {
		return nil, fmt.Errorf("no valid %s URLs provided", name)
	}

	return urls, nil
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func formatURLs(urls []*url.URL) string {
	var urlStrings []string
	for _, u := range urls {
		urlStrings = append(urlStrings, u.String())
	}
	return strings.Join(urlStrings, ", ")
}
