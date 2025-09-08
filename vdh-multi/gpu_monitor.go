package main

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

type GPUInfo struct {
	Index       int     `json:"index"`
	Name        string  `json:"name"`
	MemoryUsed  uint64  `json:"memory_used_bytes"`
	MemoryTotal uint64  `json:"memory_total_bytes"`
	MemoryUsage float64 `json:"memory_usage_percent"`
	CoreUsage   uint32  `json:"core_usage_percent"`
	Temperature uint32  `json:"temperature_celsius"`
	PowerDraw   uint32  `json:"power_draw_watts"`
}

type GPUResponse struct {
	GPUs  []GPUInfo `json:"gpus"`
	Count int       `json:"count"`
	Error string    `json:"error,omitempty"`
}

func gpuHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Initialize NVML
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		response := GPUResponse{
			Error: "Failed to initialize NVML: " + nvml.ErrorString(ret),
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(response)
		return
	}
	defer nvml.Shutdown()

	// Get device count
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		response := GPUResponse{
			Error: "Failed to get device count: " + nvml.ErrorString(ret),
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(response)
		return
	}

	var gpus []GPUInfo

	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			log.Printf("Failed to get device handle for GPU %d: %v", i, nvml.ErrorString(ret))
			continue
		}

		gpu := GPUInfo{Index: i}

		// Get GPU name
		name, ret := device.GetName()
		if ret == nvml.SUCCESS {
			gpu.Name = name
		}

		// Get memory info
		memInfo, ret := device.GetMemoryInfo()
		if ret == nvml.SUCCESS {
			gpu.MemoryUsed = memInfo.Used
			gpu.MemoryTotal = memInfo.Total
			if memInfo.Total > 0 {
				gpu.MemoryUsage = float64(memInfo.Used) / float64(memInfo.Total) * 100
			}
		}

		// Get utilization rates
		utilization, ret := device.GetUtilizationRates()
		if ret == nvml.SUCCESS {
			gpu.CoreUsage = utilization.Gpu
		}

		// Get temperature
		temp, ret := device.GetTemperature(nvml.TEMPERATURE_GPU)
		if ret == nvml.SUCCESS {
			gpu.Temperature = temp
		}

		// Get power usage
		power, ret := device.GetPowerUsage()
		if ret == nvml.SUCCESS {
			gpu.PowerDraw = power / 1000 // Convert milliwatts to watts
		}

		gpus = append(gpus, gpu)
	}

	response := GPUResponse{
		GPUs:  gpus,
		Count: len(gpus),
	}

	json.NewEncoder(w).Encode(response)
}