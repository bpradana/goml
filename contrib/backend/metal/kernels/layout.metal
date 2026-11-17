kernel void fill_values(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                        device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = p.scalars[0];
    }
}

kernel void copy_values(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                        device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[position(i, p, true)] = a[position(i, p)];
    }
}

kernel void finite_values(device const float *input [[buffer(0)]],
                          device const float *unused [[buffer(1)]],
                          device atomic_uint *status [[buffer(2)]],
                          constant Params &p [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
    if (index < p.n && !isfinite(input[position(index, p)])) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}
