package cpu

import "github.com/bpradana/goml/tensor/backend"

var (
	_ backend.Device                     = Device{}
	_ backend.HostAccess                 = Device{}
	_ backend.UnaryBackend               = Device{}
	_ backend.BinaryBackend              = Device{}
	_ backend.ScalarBackend              = Device{}
	_ backend.ReductionBackend           = Device{}
	_ backend.MatMulBackend              = Device{}
	_ backend.FillBackend                = Device{}
	_ backend.CopyBackend                = Device{}
	_ backend.ConcatBackend              = Device{}
	_ backend.ConvolutionBackend         = Device{}
	_ backend.ConvolutionBackwardBackend = Device{}
	_ backend.PatchBackend               = Device{}
	_ backend.PoolBackend                = Device{}
	_ backend.PoolBackwardBackend        = Device{}
	_ backend.ResizeBackend              = Device{}
	_ backend.ResizeBackwardBackend      = Device{}
	_ backend.FiniteBackend              = Device{}
	_ backend.SubScaledBackend           = Device{}
	_ backend.HostAdamBackend            = Device{}
)
