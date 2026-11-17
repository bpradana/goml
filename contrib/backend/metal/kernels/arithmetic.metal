kernel void binary_values(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                          device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    float x = a[position(i, p)], y = b[position(i, p, true)];
    switch (p.op) {
    case 0:
        out[i] = x + y;
        break;
    case 1:
        out[i] = x - y;
        break;
    case 2:
        out[i] = x * y;
        break;
    default:
        out[i] = x / y;
    }
}

kernel void unary_values(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                         device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    float x = a[position(i, p)], s;
    switch (p.op) {
    case 0:
        s = x * p.scalars[0];
        break;
    case 1:
        s = exp(x);
        break;
    case 2:
        s = log(x);
        break;
    case 3:
        s = sqrt(x);
        break;
    case 4:
        s = isnan(x) ? x : max(0.0f, x);
        break;
    case 5:
        s = sigmoid(x);
        break;
    case 6:
        s = tanh(x);
        break;
    case 7:
        s = (isinf(x) && x < 0) ? -0.0f : x * sigmoid(x);
        break;
    case 8:
        s = isnan(x) ? x : (x > 0 ? 1.0f : 0.0f);
        break;
    case 9:
        s = sigmoid(x);
        s = s * (1 - s);
        break;
    case 10:
        s = tanh(x);
        s = 1 - s * s;
        break;
    case 11:
        s = sigmoid(x);
        s = (isinf(x) ? (x > 0 ? 1.0f : 0.0f) : s + x * s * (1 - s));
        break;
    case 12:
        s = x + p.scalars[0];
        break;
    default:
        s = (i % p.shape[p.rank - 1] > i / p.shape[p.rank - 1]) ? p.scalars[0] : x;
    }
    out[i] = s;
}
