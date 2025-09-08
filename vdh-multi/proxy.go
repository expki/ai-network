package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
	"strings"
)

var (
	chatPattern   = regexp.MustCompile(`/(?:v1/)?(?:(?:chat/)?completions?|completion|complete|infill)/?$`)
	embedPattern  = regexp.MustCompile(`/(?:v1/)?embeddings?/?$`)
	rerankPattern = regexp.MustCompile(`/(?:v1/)?rerank(?:ing)?/?$`)
)

// BackendConfig holds the single backend URL for each model type
type backendConfig struct {
	chatURL   *url.URL
	embedURL  *url.URL
	rerankURL *url.URL
}

func createReverseProxy(chatTargets, embedTargets, rerankTargets []*url.URL) http.Handler {
	// Use first URL from each target list (should only be one now)
	var chatURL, embedURL, rerankURL *url.URL
	if len(chatTargets) > 0 {
		chatURL = chatTargets[0]
	}
	if len(embedTargets) > 0 {
		embedURL = embedTargets[0]
	}
	if len(rerankTargets) > 0 {
		rerankURL = rerankTargets[0]
	}
	
	config := &backendConfig{
		chatURL:   chatURL,
		embedURL:  embedURL,
		rerankURL: rerankURL,
	}
	
	proxy := &httputil.ReverseProxy{
		Director:       createDirector(config),
		ModifyResponse: createModifyResponse(),
		ErrorHandler:   createErrorHandler(),
	}

	statusHandler := createStatusHandler(config)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/ping":
			w.Header().Set("Content-Type", "text/plain")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("pong"))
			return
		case "/gpus":
			gpuHandler(w, r)
			return
		case "/status":
			statusHandler(w, r)
			return
		default:
			proxy.ServeHTTP(w, r)
		}
	})
}

func createDirector(config *backendConfig) func(*http.Request) {
	return func(req *http.Request) {
		backendURL := selectBackend(req, config)
		if backendURL == nil {
			log.Printf("Error: No backend available for request")
			return
		}

		// update request URL
		req.URL.Scheme = backendURL.Scheme
		req.URL.Host = backendURL.Host
		req.URL.Path = singleJoiningSlash(backendURL.Path, req.URL.Path)

		if backendURL.RawQuery == "" || req.URL.RawQuery == "" {
			req.URL.RawQuery = backendURL.RawQuery + req.URL.RawQuery
		} else {
			req.URL.RawQuery = backendURL.RawQuery + "&" + req.URL.RawQuery
		}
		if err := decompressRequest(req); err != nil {
			log.Printf("Failed to decompress request: %v", err)
		}

		// clean up headers
		req.Header.Del("Accept-Encoding") // prevent upstream compression
		req.Header.Del("Authorization")   // remove auth header before forwarding
		req.Host = backendURL.Host
	}
}

func selectBackend(req *http.Request, config *backendConfig) *url.URL {
	path := req.URL.Path
	queryType := req.Header.Get("Query-Type")

	if queryType != "" {
		req.Header.Del("Query-Type")
	}

	// Select backend based on Query-Type header or path pattern
	var backendURL *url.URL
	switch {
	case strings.EqualFold(queryType, "chat") || (queryType == "" && chatPattern.MatchString(path)):
		backendURL = config.chatURL
		if backendURL != nil {
			log.Printf("Chat query detected, routing to %s", backendURL.Host)
		}
	case strings.EqualFold(queryType, "embed") || (queryType == "" && embedPattern.MatchString(path)):
		backendURL = config.embedURL
		if backendURL != nil {
			log.Printf("Embed query detected, routing to %s", backendURL.Host)
		}
	case strings.EqualFold(queryType, "rerank") || (queryType == "" && rerankPattern.MatchString(path)):
		backendURL = config.rerankURL
		if backendURL != nil {
			log.Printf("Rerank query detected, routing to %s", backendURL.Host)
		}
	default:
		log.Println("No query detected, fallback to Chat")
		backendURL = config.chatURL
		if backendURL != nil {
			log.Printf("Routing to chat backend %s", backendURL.Host)
		}
	}

	if backendURL == nil {
		// Fallback to chat backend if available
		if config.chatURL != nil {
			backendURL = config.chatURL
			log.Printf("Warning: No backend available, using fallback: %s", backendURL.Host)
		}
	}

	return backendURL
}

func createModifyResponse() func(*http.Response) error {
	return func(resp *http.Response) error {
		if resp.Request != nil {
			// Remove Content-Length header as compression middleware will handle it
			resp.Header.Del("Content-Length")
			resp.ContentLength = -1
		}

		return nil
	}
}

func createErrorHandler() func(http.ResponseWriter, *http.Request, error) {
	return func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Proxy error: %v", err)
		http.Error(w, "Bad Gateway", http.StatusBadGateway)
	}
}


func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")

	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	default:
		return a + b
	}
}
