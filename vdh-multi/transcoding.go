package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/expki/calculator/lib/encoding"
)

func transcodeBinarytoJSON(r *http.Request) error {
	if !strings.EqualFold(r.Header.Get("Encode-Binary"), "jsonb") {
		return nil
	}
	
	data, _ := encoding.DecodeIO(r.Body)
	r.Body.Close()
	
	raw, err := json.Marshal(data)
	if err != nil {
		return errors.Join(errors.New("transcode binary to json"), err)
	}
	
	r.Body = io.NopCloser(bytes.NewBuffer(raw))
	r.ContentLength = int64(len(raw))
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(raw)))
	r.Header.Del("Encode-Binary")
	
	return nil
}

func transcodeJSONtoBinary(resp *http.Response) error {
	if resp.Request == nil || !strings.EqualFold(resp.Request.Header.Get("Accept-Binary"), "true") {
		return nil
	}
	
	bodyBytes, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return errors.Join(errors.New("read response body for json to binary transcode"), err)
	}

	var data any
	if err := json.Unmarshal(bodyBytes, &data); err != nil {
		return errors.Join(errors.New("unmarshal json for binary transcode"), err)
	}

	encoded := encoding.Encode(data)
	resp.Body = io.NopCloser(bytes.NewBuffer(encoded))
	resp.ContentLength = int64(len(encoded))
	resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(encoded)))
	resp.Header.Set("Content-Type", "application/octet-stream")
	
	return nil
}