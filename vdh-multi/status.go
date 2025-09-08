package main

import (
	"encoding/json"
	"net/http"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

type BackendStatus struct {
	URL  string `json:"url"`
	Type string `json:"type"`
}

type StatusResponse struct {
	Backends []BackendStatus `json:"backends"`
	GPUs     []GPUInfo       `json:"gpus,omitempty"`
}

func createStatusHandler(config *backendConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		var backends []BackendStatus

		// Add chat backend
		if config.chatURL != nil {
			backends = append(backends, BackendStatus{
				URL:  config.chatURL.String(),
				Type: "chat",
			})
		}

		// Add embed backend
		if config.embedURL != nil {
			backends = append(backends, BackendStatus{
				URL:  config.embedURL.String(),
				Type: "embed",
			})
		}

		// Add rerank backend
		if config.rerankURL != nil {
			backends = append(backends, BackendStatus{
				URL:  config.rerankURL.String(),
				Type: "rerank",
			})
		}

		response := StatusResponse{
			Backends: backends,
		}

		// Optionally include GPU info if available
		if r.URL.Query().Get("include_gpus") == "true" {
			gpus := getGPUInfo()
			if gpus != nil {
				response.GPUs = gpus
			}
		}

		json.NewEncoder(w).Encode(response)
	}
}

func getGPUInfo() []GPUInfo {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		return nil
	}
	defer nvml.Shutdown()

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return nil
	}

	var gpus []GPUInfo
	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			continue
		}

		gpu := GPUInfo{Index: i}

		name, ret := device.GetName()
		if ret == nvml.SUCCESS {
			gpu.Name = name
		}

		memInfo, ret := device.GetMemoryInfo()
		if ret == nvml.SUCCESS {
			gpu.MemoryUsed = memInfo.Used
			gpu.MemoryTotal = memInfo.Total
			if memInfo.Total > 0 {
				gpu.MemoryUsage = float64(memInfo.Used) / float64(memInfo.Total) * 100
			}
		}

		utilization, ret := device.GetUtilizationRates()
		if ret == nvml.SUCCESS {
			gpu.CoreUsage = utilization.Gpu
		}

		temp, ret := device.GetTemperature(nvml.TEMPERATURE_GPU)
		if ret == nvml.SUCCESS {
			gpu.Temperature = temp
		}

		power, ret := device.GetPowerUsage()
		if ret == nvml.SUCCESS {
			gpu.PowerDraw = power / 1000
		}

		gpus = append(gpus, gpu)
	}

	return gpus
}