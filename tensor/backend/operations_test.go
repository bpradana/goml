package backend

import "testing"

func TestOperations_DiagnosticNames(t *testing.T) {
	for _, test := range []struct{ got, want string }{
		{Exp.String(), "exp"},
		{SiLUGrad.String(), "silu_grad"},
		{Add.String(), "add"},
		{Div.String(), "div"},
		{Scale.String(), "scale"},
		{MaskUpperTriangle.String(), "mask_upper_triangle"},
		{Sum.String(), "sum"},
		{Max.String(), "max"},
		{Nearest.String(), "nearest"},
		{Bicubic.String(), "bicubic"},
		{UnaryOp(0).String(), "UnaryOp(0)"},
		{BinaryOp(255).String(), "BinaryOp(255)"},
		{ScalarOp(255).String(), "ScalarOp(255)"},
		{ReduceOp(255).String(), "ReduceOp(255)"},
		{ResizeMode(255).String(), "ResizeMode(255)"},
	} {
		if test.got != test.want {
			t.Fatalf("got %q, want %q", test.got, test.want)
		}
	}
}
