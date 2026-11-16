package yolo

import "slices"

func suppress(candidates []Detection, threshold float32, limit int) []Detection {
	slices.SortStableFunc(candidates, func(a, b Detection) int {
		if a.Confidence > b.Confidence {
			return -1
		}

		if a.Confidence < b.Confidence {
			return 1
		}

		return 0
	})

	if len(candidates) > 30000 {
		candidates = candidates[:30000]
	}

	selected := make([]Detection, 0, min(len(candidates), limit))
	for _, candidate := range candidates {
		overlap := false

		for _, kept := range selected {
			if candidate.ClassID == kept.ClassID && intersectionOverUnion(candidate.Box, kept.Box) > threshold {
				overlap = true
				break
			}
		}

		if !overlap {
			selected = append(selected, candidate)
			if len(selected) == limit {
				break
			}
		}
	}

	return selected
}

func intersectionOverUnion(a, b Box) float32 {
	width := max(0, min(a.Right, b.Right)-max(a.Left, b.Left))
	height := max(0, min(a.Bottom, b.Bottom)-max(a.Top, b.Top))
	intersection := width * height

	union := max(0, a.Right-a.Left)*max(0, a.Bottom-a.Top) + max(0, b.Right-b.Left)*max(0, b.Bottom-b.Top) - intersection
	if union <= 0 {
		return 0
	}

	return intersection / union
}
