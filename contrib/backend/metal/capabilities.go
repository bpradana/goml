package metal

import "github.com/bpradana/goml/tensor/backend"

var (
	_ backend.Device                     = (*Device)(nil)
	_ backend.Buffer                     = (*buffer)(nil)
	_ backend.UnaryBackend               = (*Device)(nil)
	_ backend.BinaryBackend              = (*Device)(nil)
	_ backend.ScalarBackend              = (*Device)(nil)
	_ backend.ReductionBackend           = (*Device)(nil)
	_ backend.MatMulBackend              = (*Device)(nil)
	_ backend.FillBackend                = (*Device)(nil)
	_ backend.CopyBackend                = (*Device)(nil)
	_ backend.ConvolutionBackend         = (*Device)(nil)
	_ backend.ConvolutionBackwardBackend = (*Device)(nil)
	_ backend.PoolBackend                = (*Device)(nil)
	_ backend.PoolBackwardBackend        = (*Device)(nil)
	_ backend.ResizeBackend              = (*Device)(nil)
	_ backend.ResizeBackwardBackend      = (*Device)(nil)
	_ backend.FiniteBackend              = (*Device)(nil)
)
