kernel void reduce_values(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                          device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    uint base = p.ao, remaining = i, count = 1;
    for (int axis = int(p.rank) - 1; axis >= 0; --axis) {
        if (p.flags & (1u << axis)) {
            count *= p.shape[axis];
        } else {
            uint c = remaining % p.shape[axis];
            remaining /= p.shape[axis];
            base += c * p.as[axis];
        }
    }
    float result = p.op == 2 ? -INFINITY : 0.0f;
    for (uint j = 0; j < count; ++j) {
        uint index = base, q = j;
        for (int axis = int(p.rank) - 1; axis >= 0; --axis) {
            if (p.flags & (1u << axis)) {
                index += (q % p.shape[axis]) * p.as[axis];
                q /= p.shape[axis];
            }
        }
        float x = a[index];
        if (p.op == 2) {
            result = (isnan(x) || isnan(result)) ? NAN : max(result, x);
        } else {
            result += x;
        }
    }
    out[i] = p.op == 1 ? result / float(count) : result;
}
