package backend

// UnaryOp identifies an existing element-wise numerical primitive.
type UnaryOp uint8

// Unary operations preserve shape and follow IEEE Float32 special-value rules.
// Activation gradients denote the derivative before multiplying an incoming gradient.
const (
	Exp UnaryOp = iota + 1
	Log
	Sqrt
	ReLU
	Sigmoid
	Tanh
	SiLU
	ReLUGrad
	SigmoidGrad
	TanhGrad
	SiLUGrad
)

// BinaryOp identifies trailing-dimension broadcast arithmetic.
type BinaryOp uint8

// Division follows IEEE arithmetic, including division by zero.
const (
	Add BinaryOp = iota + 1
	Sub
	Mul
	Div
)

// ScalarOp identifies a numerical operation with one host scalar argument.
type ScalarOp uint8

// MaskUpperTriangle requires a matrix and replaces entries above its diagonal.
const (
	Scale ScalarOp = iota + 1
	AddScalar
	MaskUpperTriangle
)

// ReduceOp identifies an axis reduction. Reduced dimensions are removed.
type ReduceOp uint8

// Empty sums are zero. Empty mean/max groups fail. NaNs propagate.
const (
	Sum ReduceOp = iota + 1
	Mean
	Max
)

// ResizeMode selects integer-scale interpolation with align_corners=false.
type ResizeMode uint8

// Bicubic interpolation uses cubic coefficient -0.75.
const (
	Nearest ResizeMode = iota + 1
	Bilinear
	Bicubic
)
