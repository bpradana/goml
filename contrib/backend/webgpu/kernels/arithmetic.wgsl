@compute @workgroup_size(64)
fn unary_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let value = a[index_of(i, p.a_strides, p.a_offset)];
    var result = value;
    if !is_nan(value) {
        switch p.op {
            case 1u: {
                result = exp(value);
            }
            case 2u: {
                if value < 0.0 {
                    result = nan_value();
                } else if value == 0.0 {
                    result = -infinity();
                } else {
                    result = log(value);
                }
            }
            case 3u: {
                if value < 0.0 {
                    result = nan_value();
                } else {
                    result = sqrt(value);
                }
            }
            case 4u: {
                result = max(value, 0.0);
            }
            case 5u: {
                result = sigmoid(value);
            }
            case 6u: {
                result = tanh_value(value);
            }
            case 7u: {
                result = value * sigmoid(value);
                if value == -infinity() {
                    result = bitcast<f32>(0x80000000u);
                }
            }
            case 8u: {
                result = select(0.0, 1.0, value > 0.0);
            }
            case 9u: {
                let s = sigmoid(value);
                result = s * (1.0 - s);
            }
            case 10u: {
                let t = tanh_value(value);
                result = 1.0 - t * t;
            }
            case 11u: {
                let s = sigmoid(value);
                result = s + value * s * (1.0 - s);
                if value == -infinity() {
                    result = 0.0;
                }
                if value == infinity() {
                    result = 1.0;
                }
            }
            default: {}
        }
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = result;
}

@compute @workgroup_size(64)
fn binary_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let x = a[index_of(i, p.a_strides, p.a_offset)];
    let y = b[index_of(i, p.b_strides, p.b_offset)];
    var result = 0.0;
    switch p.op {
        case 1u: {
            result = x + y;
        }
        case 2u: {
            result = x - y;
        }
        case 3u: {
            result = x * y;
            if (x == 0.0 || y == 0.0) && is_finite(x) && is_finite(y) {
                result = bitcast<f32>((bitcast<u32>(x) ^ bitcast<u32>(y)) & 0x80000000u);
            }
        }
        case 4u: {
            result = x / y;
        }
        default: {}
    }
    if is_nan(x) || is_nan(y) {
        result = nan_value();
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = result;
}

@compute @workgroup_size(64)
fn scalar_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let x = a[index_of(i, p.a_strides, p.a_offset)];
    var result = x;
    switch p.op {
        case 1u: {
            result = x * p.scalars.x;
        }
        case 2u: {
            result = x + p.scalars.x;
        }
        case 3u: {
            if i % p.shape[1] > i / p.shape[1] {
                result = p.scalars.x;
            }
        }
        default: {}
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = result;
}
