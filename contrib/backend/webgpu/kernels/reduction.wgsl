@compute @workgroup_size(64)
fn reduce_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    var coordinates: array<u32, 8>;
    var remaining = i;
    var source = p.a_offset;
    var destination = p.dst_offset;
    var out_axis = 0u;
    var reduced = 1u;
    for (var axis = 0u; axis < p.rank; axis++) {
        if (p.flags & (1u << axis)) == 0u {
            out_axis++;
        } else {
            reduced *= p.shape[axis];
        }
    }
    for (var axis = i32(p.rank) - 1; axis >= 0; axis--) {
        if (p.flags & (1u << u32(axis))) == 0u {
            out_axis--;
            coordinates[axis] = remaining % p.shape[axis];
            remaining /= p.shape[axis];
            source += coordinates[axis] * p.a_strides[axis];
            destination += coordinates[axis] * p.dst_strides[out_axis];
        }
    }

    var total = 0.0;
    if p.op == 3u {
        total = -infinity();
    }
    for (var j = 0u; j < reduced; j++) {
        var rest = j;
        var index = source;
        for (var axis = i32(p.rank) - 1; axis >= 0; axis--) {
            if (p.flags & (1u << u32(axis))) != 0u {
                index += (rest % p.shape[axis]) * p.a_strides[axis];
                rest /= p.shape[axis];
            }
        }

        let value = a[index];
        if is_nan(value) {
            total = nan_value();
            break;
        }
        if p.op == 3u {
            total = max(total, value);
        } else {
            total += value;
        }
    }
    if p.op == 2u {
        total /= f32(reduced);
    }

    dst[destination] = total;
}
