package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
)

var (
	chatPattern   = regexp.MustCompile(`/(?:v1/)?(?:(?:chat/)?completions?|completion|complete|infill)/?$`)
	embedPattern  = regexp.MustCompile(`/(?:v1/)?embeddings?/?$`)
	rerankPattern = regexp.MustCompile(`/(?:v1/)?rerank(?:ing)?/?$`)
)

// Backend represents a single backend server with connection tracking
type backend struct {
	url         *url.URL
	connections int64
	mu          sync.RWMutex
}

func (b *backend) incrementConnections() {
	atomic.AddInt64(&b.connections, 1)
}

func (b *backend) decrementConnections() {
	atomic.AddInt64(&b.connections, -1)
}

func (b *backend) getConnections() int64 {
	return atomic.LoadInt64(&b.connections)
}

// Load balancer state for least-connections selection
type loadBalancer struct {
	chatBackends   []*backend
	embedBackends  []*backend
	rerankBackends []*backend
	mu             sync.RWMutex
}

func newLoadBalancer(chatTargets, embedTargets, rerankTargets []*url.URL) *loadBalancer {
	lb := &loadBalancer{
		chatBackends:   make([]*backend, len(chatTargets)),
		embedBackends:  make([]*backend, len(embedTargets)),
		rerankBackends: make([]*backend, len(rerankTargets)),
	}

	for i, url := range chatTargets {
		lb.chatBackends[i] = &backend{url: url}
	}
	for i, url := range embedTargets {
		lb.embedBackends[i] = &backend{url: url}
	}
	for i, url := range rerankTargets {
		lb.rerankBackends[i] = &backend{url: url}
	}

	return lb
}

func (lb *loadBalancer) getLeastConnectedBackend(backends []*backend) *backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *backend
	minConnections := int64(^uint64(0) >> 1) // Max int64

	for _, b := range backends {
		connections := b.getConnections()
		if connections < minConnections {
			minConnections = connections
			selected = b
		}
	}

	return selected
}

func (lb *loadBalancer) getNextChat() *backend {
	return lb.getLeastConnectedBackend(lb.chatBackends)
}

func (lb *loadBalancer) getNextEmbed() *backend {
	return lb.getLeastConnectedBackend(lb.embedBackends)
}

func (lb *loadBalancer) getNextRerank() *backend {
	return lb.getLeastConnectedBackend(lb.rerankBackends)
}

func createReverseProxy(chatTargets, embedTargets, rerankTargets []*url.URL) http.Handler {
	lb := newLoadBalancer(chatTargets, embedTargets, rerankTargets)
	
	proxy := &httputil.ReverseProxy{
		Director:       createDirector(lb),
		ModifyResponse: createModifyResponse(lb),
		ErrorHandler:   createErrorHandler(lb),
	}

	statusHandler := createStatusHandler(lb)

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

func createDirector(lb *loadBalancer) func(*http.Request) {
	return func(req *http.Request) {
		backend := selectBackend(req, lb)
		if backend == nil {
			log.Printf("Error: No backend available for request")
			return
		}

		// Increment connection count for this backend
		backend.incrementConnections()
		
		// Store backend in request context so we can decrement later
		req.Header.Set("X-Backend-URL", backend.url.String())

		// update request URL
		req.URL.Scheme = backend.url.Scheme
		req.URL.Host = backend.url.Host
		req.URL.Path = singleJoiningSlash(backend.url.Path, req.URL.Path)

		if backend.url.RawQuery == "" || req.URL.RawQuery == "" {
			req.URL.RawQuery = backend.url.RawQuery + req.URL.RawQuery
		} else {
			req.URL.RawQuery = backend.url.RawQuery + "&" + req.URL.RawQuery
		}
		if err := decompressRequest(req); err != nil {
			log.Printf("Failed to decompress request: %v", err)
		}

		// clean up headers
		req.Header.Del("Accept-Encoding") // prevent upstream compression
		req.Header.Del("Authorization")   // remove auth header before forwarding
		req.Host = backend.url.Host
	}
}

func selectBackend(req *http.Request, lb *loadBalancer) *backend {
	path := req.URL.Path
	queryType := req.Header.Get("Query-Type")

	if queryType != "" {
		req.Header.Del("Query-Type")
	}

	// Select backend based on Query-Type header or path pattern with least-connections
	var backend *backend
	switch {
	case strings.EqualFold(queryType, "chat") || (queryType == "" && chatPattern.MatchString(path)):
		backend = lb.getNextChat()
		if backend != nil {
			log.Printf("Chat query detected, routing to %s (connections: %d)", backend.url.Host, backend.getConnections())
		}
	case strings.EqualFold(queryType, "embed") || (queryType == "" && embedPattern.MatchString(path)):
		backend = lb.getNextEmbed()
		if backend != nil {
			log.Printf("Embed query detected, routing to %s (connections: %d)", backend.url.Host, backend.getConnections())
		}
	case strings.EqualFold(queryType, "rerank") || (queryType == "" && rerankPattern.MatchString(path)):
		backend = lb.getNextRerank()
		if backend != nil {
			log.Printf("Rerank query detected, routing to %s (connections: %d)", backend.url.Host, backend.getConnections())
		}
	default:
		log.Println("No query detected, fallback to Chat")
		backend = lb.getNextChat()
		if backend != nil {
			log.Printf("Routing to chat backend %s (connections: %d)", backend.url.Host, backend.getConnections())
		}
	}

	if backend == nil {
		// Fallback to first chat backend if available
		if len(lb.chatBackends) > 0 {
			backend = lb.chatBackends[0]
			log.Printf("Warning: No backend available, using fallback: %s", backend.url.Host)
		}
	}

	return backend
}

func createModifyResponse(lb *loadBalancer) func(*http.Response) error {
	return func(resp *http.Response) error {
		// Decrement connection count for the backend that handled this request
		if resp.Request != nil {
			backendURL := resp.Request.Header.Get("X-Backend-URL")
			if backendURL != "" {
				decrementBackendConnection(lb, backendURL)
				resp.Request.Header.Del("X-Backend-URL")
			}
			
			// Remove Content-Length header as compression middleware will handle it
			resp.Header.Del("Content-Length")
			resp.ContentLength = -1
		}

		return nil
	}
}

func createErrorHandler(lb *loadBalancer) func(http.ResponseWriter, *http.Request, error) {
	return func(w http.ResponseWriter, r *http.Request, err error) {
		// Decrement connection count on error
		backendURL := r.Header.Get("X-Backend-URL")
		if backendURL != "" {
			decrementBackendConnection(lb, backendURL)
			r.Header.Del("X-Backend-URL")
		}
		
		log.Printf("Proxy error: %v", err)
		http.Error(w, "Bad Gateway", http.StatusBadGateway)
	}
}

func decrementBackendConnection(lb *loadBalancer, backendURL string) {
	// Find and decrement the backend connection count
	for _, b := range lb.chatBackends {
		if b.url.String() == backendURL {
			b.decrementConnections()
			return
		}
	}
	for _, b := range lb.embedBackends {
		if b.url.String() == backendURL {
			b.decrementConnections()
			return
		}
	}
	for _, b := range lb.rerankBackends {
		if b.url.String() == backendURL {
			b.decrementConnections()
			return
		}
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
