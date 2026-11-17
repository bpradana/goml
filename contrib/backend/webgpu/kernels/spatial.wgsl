fn spatial_index(strides: array<u32, 8>, offset: u32, batch: u32, channel: u32, y: u32, x: u32) -> u32 {
    return offset + batch * strides[0] + channel * strides[1] + y * strides[2] + x * strides[3];
}

fn pool_winner(batch: u32, channel: u32, y: u32, x: u32) -> i32 {
    var winner = -1;
    var best = -infinity();
    for (var ky = 0u; ky < p.ints[4]; ky++) {
        let iy = i32(y * p.ints[5] + ky) - i32(p.ints[6]);
        if iy < 0 || iy >= i32(p.ints[0]) {
            continue;
        }
        for (var kx = 0u; kx < p.ints[4]; kx++) {
            let ix = i32(x * p.ints[5] + kx) - i32(p.ints[6]);
            if ix < 0 || ix >= i32(p.ints[1]) {
                continue;
            }

            let value = a[spatial_index(p.a_strides, p.a_offset, batch, channel, u32(iy), u32(ix))];
            if winner < 0 || (!is_nan(best) && (is_nan(value) || value > best)) {
                best = value;
                winner = iy * i32(p.ints[1]) + ix;
            }
        }
    }

    return winner;
}

@compute @workgroup_size(64)
fn pool_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let x = i % p.ints[3];
    let y = (i / p.ints[3]) % p.ints[2];
    let channel = (i / (p.ints[3] * p.ints[2])) % p.shape[1];
    let batch = i / (p.ints[3] * p.ints[2] * p.shape[1]);
    let winner = pool_winner(batch, channel, y, x);
    var value = -infinity();
    if winner >= 0 {
        value = a[spatial_index(p.a_strides, p.a_offset, batch, channel, u32(winner) / p.ints[1], u32(winner) % p.ints[1])];
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = value;
}

@compute @workgroup_size(64)
fn pool_gradients(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let x = i % p.ints[1];
    let y = (i / p.ints[1]) % p.ints[0];
    let channel = (i / (p.ints[1] * p.ints[0])) % p.shape[1];
    let batch = i / (p.ints[1] * p.ints[0] * p.shape[1]);
    let stride = i32(p.ints[5]);
    let sy = max(0, (i32(y + p.ints[6]) - i32(p.ints[4]) + stride) / stride);
    let sx = max(0, (i32(x + p.ints[6]) - i32(p.ints[4]) + stride) / stride);
    let ey = min(i32(p.ints[2]) - 1, i32(y + p.ints[6]) / stride);
    let ex = min(i32(p.ints[3]) - 1, i32(x + p.ints[6]) / stride);
    var total = 0.0;
    for (var oy = sy; oy <= ey; oy++) {
        for (var ox = sx; ox <= ex; ox++) {
            if pool_winner(batch, channel, u32(oy), u32(ox)) == i32(y * p.ints[1] + x) {
                total += b[spatial_index(p.b_strides, p.b_offset, batch, channel, u32(oy), u32(ox))];
            }
        }
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = total;
}

fn cubic_weight(distance: f32) -> f32 {
    let x = abs(distance);
    if x <= 1.0 {
        return ((1.25 * x - 2.25) * x) * x + 1.0;
    }
    if x < 2.0 {
        return ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0;
    }

    return 0.0;
}

fn resize_sample(batch: u32, channel: u32, y: i32, x: i32) -> f32 {
    return a[spatial_index(p.a_strides, p.a_offset, batch, channel, u32(clamp(y, 0, i32(p.ints[0]) - 1)), u32(clamp(x, 0, i32(p.ints[1]) - 1)))];
}

@compute @workgroup_size(64)
fn resize_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let x = i % p.ints[3];
    let y = (i / p.ints[3]) % p.ints[2];
    let channel = (i / (p.ints[3] * p.ints[2])) % p.shape[1];
    let batch = i / (p.ints[3] * p.ints[2] * p.shape[1]);
    var total = 0.0;
    if p.ints[5] == 1u {
        total = resize_sample(batch, channel, i32(y / p.ints[4]), i32(x / p.ints[4]));
    } else {
        let fy = (f32(y) + 0.5) / f32(p.ints[4]) - 0.5;
        let fx = (f32(x) + 0.5) / f32(p.ints[4]) - 0.5;
        let iy = i32(floor(fy));
        let ix = i32(floor(fx));
        if p.ints[5] == 2u {
            let wy = fy - f32(iy);
            let wx = fx - f32(ix);
            total = (resize_sample(batch, channel, iy, ix) * (1.0 - wx) + resize_sample(batch, channel, iy, ix + 1) * wx) * (1.0 - wy)
                  + (resize_sample(batch, channel, iy + 1, ix) * (1.0 - wx) + resize_sample(batch, channel, iy + 1, ix + 1) * wx) * wy;
        } else {
            for (var ky = -1; ky <= 2; ky++) {
                for (var kx = -1; kx <= 2; kx++) {
                    total += resize_sample(batch, channel, iy + ky, ix + kx) * cubic_weight(fy - f32(iy + ky)) * cubic_weight(fx - f32(ix + kx));
                }
            }
        }
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = total;
}

fn axis_weight(output: u32, input: i32, size: u32) -> f32 {
    if p.ints[5] == 1u {
        return select(0.0, 1.0, i32(output / p.ints[4]) == input);
    }

    let coordinate = (f32(output) + 0.5) / f32(p.ints[4]) - 0.5;
    let base = i32(floor(coordinate));
    var weight = 0.0;
    if p.ints[5] == 2u {
        let fraction = coordinate - f32(base);
        if clamp(base, 0, i32(size) - 1) == input {
            weight += 1.0 - fraction;
        }
        if clamp(base + 1, 0, i32(size) - 1) == input {
            weight += fraction;
        }
    } else {
        for (var step = -1; step <= 2; step++) {
            if clamp(base + step, 0, i32(size) - 1) == input {
                weight += cubic_weight(coordinate - f32(base + step));
            }
        }
    }

    return weight;
}

@compute @workgroup_size(64)
fn resize_gradients(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let x = i % p.ints[1];
    let y = (i / p.ints[1]) % p.ints[0];
    let channel = (i / (p.ints[1] * p.ints[0])) % p.shape[1];
    let batch = i / (p.ints[1] * p.ints[0] * p.shape[1]);
    let scale = i32(p.ints[4]);
    let sy = max(0, (i32(y) - 3) * scale);
    let sx = max(0, (i32(x) - 3) * scale);
    let ey = min(i32(p.ints[2]), (i32(y) + 4) * scale);
    let ex = min(i32(p.ints[3]), (i32(x) + 4) * scale);
    var total = 0.0;
    for (var oy = sy; oy < ey; oy++) {
        let wy = axis_weight(u32(oy), i32(y), p.ints[0]);
        if wy == 0.0 {
            continue;
        }
        for (var ox = sx; ox < ex; ox++) {
            let weight = wy * axis_weight(u32(ox), i32(x), p.ints[1]);
            if weight != 0.0 {
                total += a[spatial_index(p.a_strides, p.a_offset, batch, channel, u32(oy), u32(ox))] * weight;
            }
        }
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = total;
}
