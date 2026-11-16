#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n, rank, op, flags;
    uint shape[8], as[8], bs[8];
    uint ao, bo;
    uint ints[8];
    float scalars[4];
};

uint position(uint i, constant Params &p, bool second = false) {
    uint out = second ? p.bo : p.ao;
    for (int axis = int(p.rank) - 1; axis >= 0; --axis) {
        uint c = i % p.shape[axis];
        i /= p.shape[axis];
        out += c * (second ? p.bs[axis] : p.as[axis]);
    }
    return out;
}

float sigmoid(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + exp(-x));
    }
    float e = exp(x);
    return e / (1.0f + e);
}

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

uint spatial_index(uint plane, uint y, uint x, constant Params &p, bool gradient = false) {
    uint batch = plane / p.shape[1];
    uint channel = plane % p.shape[1];
    if (gradient) {
        return p.bo + batch * p.bs[0] + channel * p.bs[1] + y * p.bs[2] + x * p.bs[3];
    }
    return p.ao + batch * p.as[0] + channel * p.as[1] + y * p.as[2] + x * p.as[3];
}

uint pool_selection(device const float *input, uint plane, uint row, uint column,
                    constant Params &p, thread float &best) {
    uint selected = 0xffffffffu;
    best = -INFINITY;
    for (uint ky = 0; ky < p.ints[0]; ++ky) {
        for (uint kx = 0; kx < p.ints[0]; ++kx) {
            int y = int(row * p.ints[1] + ky) - int(p.ints[2]);
            int x = int(column * p.ints[1] + kx) - int(p.ints[2]);
            if (y < 0 || y >= int(p.shape[2]) || x < 0 || x >= int(p.shape[3])) {
                continue;
            }
            float value = input[spatial_index(plane, uint(y), uint(x), p)];
            if (!isnan(best) && (selected == 0xffffffffu || value > best || isnan(value))) {
                best = value;
                selected = (plane * p.shape[2] + uint(y)) * p.shape[3] + uint(x);
            }
        }
    }
    return selected;
}

kernel void pool_values(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                        device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    uint area = p.ints[3] * p.ints[4];
    float best;
    pool_selection(a, i / area, (i % area) / p.ints[4], i % p.ints[4], p, best);
    out[i] = best;
}

kernel void pool_gradients(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                           device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    uint plane = i / (p.shape[2] * p.shape[3]);
    int y = int((i / p.shape[3]) % p.shape[2]);
    int x = int(i % p.shape[3]);
    int windowWidth = int(p.ints[0]), stride = int(p.ints[1]), pad = int(p.ints[2]);
    int firstY = max(0, (y + pad - windowWidth + stride) / stride);
    int lastY = min(int(p.ints[3]) - 1, (y + pad) / stride);
    int firstX = max(0, (x + pad - windowWidth + stride) / stride);
    int lastX = min(int(p.ints[4]) - 1, (x + pad) / stride);
    float result = 0;
    for (int row = firstY; row <= lastY; ++row) {
        for (int column = firstX; column <= lastX; ++column) {
            float best;
            uint selected = pool_selection(a, plane, uint(row), uint(column), p, best);
            if (selected == i) {
                result += b[spatial_index(plane, uint(row), uint(column), p, true)];
            }
        }
    }
    out[i] = result;
}

struct InterpolationPoint {
    int indices[4];
    float weights[4];
    uint count;
};

InterpolationPoint interpolation_point(uint destination, uint size, uint scale, uint mode) {
    InterpolationPoint point;
    if (mode == 0) {
        point.indices[0] = int(destination / scale);
        point.weights[0] = 1;
        point.count = 1;
        return point;
    }
    float source = (float(destination) + 0.5f) / float(scale) - 0.5f;
    int base = int(floor(source));
    if (mode == 1) {
        point.count = 2;
        point.indices[0] = clamp(base, 0, int(size) - 1);
        point.indices[1] = clamp(base + 1, 0, int(size) - 1);
        point.weights[1] = source - float(base);
        point.weights[0] = 1 - point.weights[1];
        return point;
    }
    point.count = 4;
    for (uint j = 0; j < 4; ++j) {
        int index = base + int(j) - 1;
        point.indices[j] = clamp(index, 0, int(size) - 1);
        float distance = abs(source - float(index));
        float weight = 0;
        if (distance <= 1) {
            weight = (1.25f * distance - 2.25f) * distance * distance + 1;
        } else if (distance < 2) {
            weight = ((-0.75f * distance + 3.75f) * distance - 6) * distance + 3;
        }
        point.weights[j] = weight;
    }
    return point;
}

kernel void resize_values(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                          device float *out [[buffer(2)]], constant Params &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    uint plane = i / (p.ints[0] * p.ints[1]);
    uint row = (i / p.ints[1]) % p.ints[0], column = i % p.ints[1];
    InterpolationPoint ys = interpolation_point(row, p.shape[2], p.ints[2], p.ints[3]);
    InterpolationPoint xs = interpolation_point(column, p.shape[3], p.ints[2], p.ints[3]);
    if (p.ints[3] == 0) {
        out[i] = a[spatial_index(plane, uint(ys.indices[0]), uint(xs.indices[0]), p)];
        return;
    }
    float result = 0;
    for (uint y = 0; y < ys.count; ++y) {
        for (uint x = 0; x < xs.count; ++x) {
            float weight = ys.weights[y] * xs.weights[x];
            if (weight != 0) {
                result +=
                    weight * a[spatial_index(plane, uint(ys.indices[y]), uint(xs.indices[x]), p)];
            }
        }
    }
    out[i] = result;
}

kernel void resize_gradients(device const float *a [[buffer(0)]],
                             device const float *b [[buffer(1)]], device float *out [[buffer(2)]],
                             constant Params &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    uint plane = i / (p.shape[2] * p.shape[3]);
    int y = int((i / p.shape[3]) % p.shape[2]), x = int(i % p.shape[3]);
    int scale = int(p.ints[2]);
    int firstY = max(0, (y - 3) * scale), lastY = min(int(p.ints[0]), (y + 4) * scale);
    int firstX = max(0, (x - 3) * scale), lastX = min(int(p.ints[1]), (x + 4) * scale);
    float result = 0;
    for (int row = firstY; row < lastY; ++row) {
        InterpolationPoint ys = interpolation_point(uint(row), p.shape[2], p.ints[2], p.ints[3]);
        float wy = 0;
        for (uint j = 0; j < ys.count; ++j) {
            if (ys.indices[j] == y) {
                wy += ys.weights[j];
            }
        }
        if (wy == 0) {
            continue;
        }
        for (int column = firstX; column < lastX; ++column) {
            InterpolationPoint xs =
                interpolation_point(uint(column), p.shape[3], p.ints[2], p.ints[3]);
            float wx = 0;
            for (uint j = 0; j < xs.count; ++j) {
                if (xs.indices[j] == x) {
                    wx += xs.weights[j];
                }
            }
            float weight = wx * wy;
            if (weight != 0) {
                result += weight * a[spatial_index(plane, uint(row), uint(column), p)];
            }
        }
    }
    out[i] = result;
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
