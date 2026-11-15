package shape

import "fmt"

// Broadcast returns the trailing-dimension broadcast shape, or a compatibility error.
func Broadcast(a, b []int) ([]int, error) {
	shape := make([]int, max(len(a), len(b)))
	for i := 1; i <= len(shape); i++ {
		x, y := 1, 1
		if i <= len(a) {
			x = a[len(a)-i]
		}

		if i <= len(b) {
			y = b[len(b)-i]
		}

		switch {
		case x == y:
			shape[len(shape)-i] = x
		case x == 1:
			shape[len(shape)-i] = y
		case y == 1:
			shape[len(shape)-i] = x
		default:
			return nil, fmt.Errorf("incompatible shapes %v and %v", a, b)
		}
	}

	return shape, nil
}
