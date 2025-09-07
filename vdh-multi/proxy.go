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

func createReverseProxy(chatTarget, embedTarget, rerankTarget *url.URL) http.Handler {
	proxy := &httputil.ReverseProxy{
		Director:       createDirector(chatTarget, embedTarget, rerankTarget),
		ModifyResponse: modifyResponse,
		ErrorHandler:   handleProxyError,
	}
	
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ping" {
			w.Header().Set("Content-Type", "text/plain")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("pong"))
			return
		}
		proxy.ServeHTTP(w, r)
	})
}

func createDirector(chatTarget, embedTarget, rerankTarget *url.URL) func(*http.Request) {
	return func(req *http.Request) {
		target := selectTarget(req, chatTarget, embedTarget, rerankTarget)

		// update request URL
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.URL.Path = singleJoiningSlash(target.Path, req.URL.Path)

		if target.RawQuery == "" || req.URL.RawQuery == "" {
			req.URL.RawQuery = target.RawQuery + req.URL.RawQuery
		} else {
			req.URL.RawQuery = target.RawQuery + "&" + req.URL.RawQuery
		}

		// process request: decompress -> transcode
		if err := decompressRequest(req); err != nil {
			log.Printf("Failed to decompress request: %v", err)
		}
		if err := transcodeBinarytoJSON(req); err != nil {
			log.Printf("Failed to transcode binary to JSON: %v", err)
		}

		// clean up headers
		req.Header.Del("Accept-Encoding") // prevent upstream compression
		req.Header.Del("Authorization")   // remove auth header before forwarding
		req.Host = target.Host
	}
}

func selectTarget(req *http.Request, chatTarget, embedTarget, rerankTarget *url.URL) *url.URL {
	path := req.URL.Path
	queryType := req.Header.Get("Query-Type")

	if queryType != "" {
		req.Header.Del("Query-Type")
	}

	// Select target based on Query-Type header or path pattern
	switch {
	case strings.EqualFold(queryType, "chat") || (queryType == "" && chatPattern.MatchString(path)):
		log.Println("Chat query detected")
		return chatTarget
	case strings.EqualFold(queryType, "embed") || (queryType == "" && embedPattern.MatchString(path)):
		log.Println("Embed query detected")
		return embedTarget
	case strings.EqualFold(queryType, "rerank") || (queryType == "" && rerankPattern.MatchString(path)):
		log.Println("Rerank query detected")
		return rerankTarget
	default:
		log.Println("No query detected, fallback to Chat")
		return chatTarget
	}
}

func modifyResponse(resp *http.Response) error {
	// Convert JSON response to binary if client requested it
	if err := transcodeJSONtoBinary(resp); err != nil {
		log.Printf("Failed to transcode JSON to binary: %v", err)
	}

	// Remove Content-Length header as compression middleware will handle it
	if resp.Request != nil {
		resp.Header.Del("Content-Length")
		resp.ContentLength = -1
	}

	return nil
}

func handleProxyError(w http.ResponseWriter, r *http.Request, err error) {
	log.Printf("Proxy error: %v", err)
	http.Error(w, "Bad Gateway", http.StatusBadGateway)
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
