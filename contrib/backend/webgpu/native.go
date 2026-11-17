//go:build cgo && !android && !ios && (darwin || linux || windows) && (amd64 || arm64)

package webgpu

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
	"github.com/oliverbestmann/webgpu/wgpu"
)

type nativeBuffer struct {
	handle *wgpu.Buffer
	size   uint64
}

type nativeDevice struct {
	instance    *wgpu.Instance
	adapter     *wgpu.Adapter
	device      *wgpu.Device
	queue       *wgpu.Queue
	layout      *wgpu.BindGroupLayout
	pipelines   map[string]*wgpu.ComputePipeline
	dummy       *wgpu.Buffer
	encoder     *wgpu.CommandEncoder
	temporaries []*wgpu.Buffer
	retired     []*nativeBuffer
	failure     *failureState
	limits      wgpu.Limits
	counts      Stats
	inFlight    bool
	encoded     int
}

func openNative(index int) (_ *nativeDevice, _ string, err error) {
	n := &nativeDevice{failure: &failureState{}, pipelines: make(map[string]*wgpu.ComputePipeline)}

	defer func() {
		if err != nil {
			n.close()
		}
	}()

	n.instance = wgpu.CreateInstance(nil)
	if !n.instance.IsValid() {
		return nil, "", fmt.Errorf("webgpu: %w: native instance unavailable", backend.ErrUnsupported)
	}

	if index == -1 {
		n.adapter, err = n.instance.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance})
	} else {
		adapters := n.instance.EnumerateAdapters(nil)
		for i, adapter := range adapters {
			if i == index {
				n.adapter = adapter
			} else {
				adapter.Release()
			}
		}

		if n.adapter == nil {
			err = fmt.Errorf("webgpu: %w: adapter index %d is unavailable", backend.ErrUnsupported, index)
		}
	}

	if err != nil {
		return nil, "", fmt.Errorf("webgpu: %w: %v", backend.ErrUnsupported, err)
	}

	info := n.adapter.GetInfo()
	if info.AdapterType == wgpu.AdapterTypeCPU {
		return nil, "", fmt.Errorf("webgpu: %w: software adapter %s", backend.ErrUnsupported, info.Device)
	}

	failure := n.failure

	n.device, err = n.adapter.RequestDevice(&wgpu.DeviceDescriptor{
		Label: "goml WebGPU",
		DeviceLostCallback: func(reason wgpu.DeviceLostReason, message string) {
			if reason != wgpu.DeviceLostReasonDestroyed {
				failure.set(fmt.Errorf("webgpu: device lost: %s: %s", reason, message))
			}
		},
	})
	if err != nil {
		return nil, "", fmt.Errorf("webgpu: request device: %w", err)
	}

	n.queue = n.device.GetQueue()

	n.limits = n.device.GetLimits()
	if n.limits.MaxStorageBuffersPerShaderStage < 5 || n.limits.MaxComputeInvocationsPerWorkgroup < 256 {
		return nil, "", fmt.Errorf("webgpu: %w: five storage bindings and 256 workgroup invocations required", backend.ErrUnsupported)
	}

	entries := make([]wgpu.BindGroupLayoutEntry, 5)
	for i := range entries {
		kind := wgpu.BufferBindingTypeReadOnlyStorage
		if i == 0 {
			kind = wgpu.BufferBindingTypeStorage
		}

		entries[i] = wgpu.BindGroupLayoutEntry{Binding: uint32(i), Visibility: wgpu.ShaderStageCompute, Buffer: wgpu.BufferBindingLayout{Type: kind}}
	}

	n.layout, err = n.device.TryCreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Entries: entries})
	if err != nil {
		return nil, "", err
	}

	pipelineLayout, err := n.device.TryCreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*wgpu.BindGroupLayout{n.layout}})
	if err != nil {
		return nil, "", err
	}
	defer pipelineLayout.Release()

	common, err := shaders.ReadFile("kernels/common.wgsl")
	if err != nil {
		return nil, "", err
	}

	for _, source := range []struct {
		file    string
		entries []string
	}{
		{"arithmetic", []string{"unary_values", "binary_values", "scalar_values"}},
		{"layout", []string{"copy_values", "fill_values", "finite_values"}},
		{"matmul", []string{"matmul_values"}},
		{"reduction", []string{"reduce_values"}},
		{"convolution", []string{"conv_values", "conv_input_gradients", "conv_weight_gradients"}},
		{"spatial", []string{"pool_values", "pool_gradients", "resize_values", "resize_gradients"}},
	} {
		code, readErr := shaders.ReadFile("kernels/" + source.file + ".wgsl")
		if readErr != nil {
			return nil, "", readErr
		}

		module, createErr := n.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: source.file, WGSLSource: &wgpu.ShaderSourceWGSL{Code: string(common) + "\n" + string(code)}})
		if createErr != nil {
			return nil, "", fmt.Errorf("webgpu: compile %s: %w", source.file, createErr)
		}

		for _, entry := range source.entries {
			pipeline, createErr := n.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: entry, Layout: pipelineLayout, Compute: wgpu.ProgrammableStageDescriptor{Module: module, EntryPoint: entry}})
			if createErr != nil {
				module.Release()
				return nil, "", fmt.Errorf("webgpu: pipeline %s: %w", entry, createErr)
			}

			n.pipelines[entry] = pipeline
		}

		module.Release()
	}

	n.dummy, err = n.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: 4, Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, "", err
	}

	name := info.Device
	if name == "" {
		name = info.Description
	}

	return n, name + " (" + info.BackendType.String() + ")", nil
}

func (n *nativeDevice) allocate(size uint64) (*nativeBuffer, error) {
	if size > min(n.limits.MaxBufferSize, n.limits.MaxStorageBufferBindingSize) {
		return nil, fmt.Errorf("requested %d bytes exceeds device storage limit", size)
	}

	handle, err := n.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}

	n.counts.LiveBuffers++
	n.counts.LiveBytes += size
	n.counts.PeakBytes = max(n.counts.PeakBytes, n.counts.LiveBytes)

	return &nativeBuffer{handle: handle, size: size}, nil
}

func (n *nativeDevice) retire(b *nativeBuffer) {
	if n.encoder != nil || n.inFlight {
		n.retired = append(n.retired, b)
	} else {
		n.release(b)
	}
}

func (n *nativeDevice) release(b *nativeBuffer) {
	b.handle.Release()

	n.counts.LiveBuffers--
	n.counts.LiveBytes -= b.size
}

func (n *nativeDevice) upload(b *nativeBuffer, values []float32) error {
	if err := n.submit(); err != nil {
		return err
	}

	if len(values) == 0 {
		return nil
	}

	if err := n.queue.TryWriteBuffer(b.handle, 0, wgpu.ToBytes(values)); err != nil {
		return fmt.Errorf("webgpu: upload: %w", err)
	}

	n.inFlight = true

	return nil
}

func (n *nativeDevice) download(b *nativeBuffer, values []float32) error {
	if len(values) == 0 {
		return n.synchronize()
	}

	if err := n.submit(); err != nil {
		return err
	}

	size := uint64(len(values)) * 4

	staging, err := n.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return err
	}
	defer staging.Release()

	encoder, err := n.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer encoder.Release()

	if err := encoder.TryCopyBufferToBuffer(b.handle, 0, staging, 0, size); err != nil {
		return err
	}

	commands, err := encoder.TryFinish(nil)
	if err != nil {
		return err
	}

	n.queue.Submit(commands)
	commands.Release()

	n.inFlight = true

	mapped := make(chan wgpu.MapAsyncStatus, 1)

	if err := staging.TryMapAsync(wgpu.MapModeRead, 0, size, func(status wgpu.MapAsyncStatus) { mapped <- status }); err != nil {
		return err
	}

	n.device.Poll(true, nil)

	if status := <-mapped; status != wgpu.MapAsyncStatusSuccess {
		return fmt.Errorf("webgpu: map readback: %s", status)
	}

	copy(values, wgpu.FromBytes[float32](staging.GetMappedRange(0, uint(size))))
	err = staging.TryUnmap()
	n.inFlight = false
	n.reap()

	if err != nil {
		return err
	}

	return n.failure.get()
}

func (n *nativeDevice) dispatch(name string, p kernelParams, handles [4]*nativeBuffer, x, y uint32) error {
	if x == 0 {
		groups := (uint64(p.n) + 63) / 64
		x = uint32(min(groups, uint64(n.limits.MaxComputeWorkgroupsPerDimension)))
		y = uint32((groups + uint64(x) - 1) / uint64(x))
	}

	if x > n.limits.MaxComputeWorkgroupsPerDimension || y > n.limits.MaxComputeWorkgroupsPerDimension {
		return fmt.Errorf("webgpu: %w: workgroup grid exceeds device limits", backend.ErrUnsupported)
	}

	data, err := p.bytes()
	if err != nil {
		return err
	}

	arguments, err := n.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: data, Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return err
	}

	entries := make([]wgpu.BindGroupEntry, 5)

	for i, handle := range handles {
		value := n.dummy
		if handle != nil {
			value = handle.handle
		}

		entries[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: value, Size: wgpu.WholeSize}
	}

	entries[4] = wgpu.BindGroupEntry{Binding: 4, Buffer: arguments, Size: wgpu.WholeSize}

	group, err := n.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: n.layout, Entries: entries})
	if err != nil {
		arguments.Release()
		return err
	}
	defer group.Release()

	if n.encoder == nil {
		n.encoder, err = n.device.TryCreateCommandEncoder(nil)
		if err != nil {
			arguments.Release()
			return err
		}
	}

	n.temporaries = append(n.temporaries, arguments)
	pass := n.encoder.BeginComputePass(nil)
	pass.SetPipeline(n.pipelines[name])
	pass.SetBindGroup(0, group, nil)
	pass.DispatchWorkgroups(x, y, 1)
	err = pass.TryEnd()
	pass.Release()

	if err != nil {
		n.failure.set(err)
		return err
	}

	n.encoded++
	if n.encoded >= 64 {
		return n.submit()
	}

	return n.failure.get()
}

func (n *nativeDevice) submit() error {
	if n.encoder != nil {
		commands, err := n.encoder.TryFinish(nil)
		n.encoder.Release()
		n.encoder = nil

		n.encoded = 0
		if err != nil {
			n.failure.set(err)
			return err
		}

		n.queue.Submit(commands)
		commands.Release()

		n.inFlight = true
	} else if n.inFlight {
		n.queue.Submit()
	}

	return n.failure.get()
}

func (n *nativeDevice) synchronize() error {
	submitErr := n.submit()

	if n.inFlight {
		// Flush queued host writes even when an earlier encoding failed. Close
		// must drain valid submissions before releasing their physical buffers.
		n.queue.Submit()

		done := make(chan wgpu.QueueWorkDoneStatus, 1)

		n.queue.OnSubmittedWorkDone(func(status wgpu.QueueWorkDoneStatus) { done <- status })
		n.device.Poll(true, nil)

		if status := <-done; status != wgpu.QueueWorkDoneStatusSuccess {
			n.failure.set(fmt.Errorf("webgpu: command completion: %s", status))
		}

		n.inFlight = false
	}

	n.reap()

	if submitErr != nil {
		return submitErr
	}

	return n.failure.get()
}

func (n *nativeDevice) reap() {
	for _, b := range n.retired {
		n.release(b)
	}

	for _, temporary := range n.temporaries {
		temporary.Release()
	}

	n.retired = nil
	n.temporaries = nil
}

func (n *nativeDevice) stats() Stats {
	return n.counts
}

func (n *nativeDevice) close() {
	if n.encoder != nil {
		n.encoder.Release()
	}

	n.reap()

	for _, pipeline := range n.pipelines {
		pipeline.Release()
	}

	if n.dummy != nil {
		n.dummy.Release()
	}

	if n.layout != nil {
		n.layout.Release()
	}

	if n.queue != nil {
		n.queue.Release()
	}

	if n.device != nil {
		n.device.Release()
	}

	if n.adapter != nil {
		n.adapter.Release()
	}

	if n.instance != nil {
		n.instance.Release()
	}
}
