fn input_index(batch: u32, channel: u32, y: u32, x: u32) -> u32 {
    return p.a_offset + batch * p.a_strides[0] + channel * p.a_strides[1] + y * p.a_strides[2] + x * p.a_strides[3];
}

fn gradient_index(batch: u32, channel: u32, y: u32, x: u32) -> u32 {
    return p.c_offset + batch * p.c_strides[0] + channel * p.c_strides[1] + y * p.c_strides[2] + x * p.c_strides[3];
}

@compute @workgroup_size(64)
fn conv_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let width = p.ints[6];
    let height = p.ints[5];
    let out_channels = p.ints[4];
    let in_channels = p.ints[1];
    let kernel = p.ints[7];
    let stride = p.ints[8];
    let padding = i32(p.ints[9]);
    let x = i % width;
    let y = (i / width) % height;
    let channel = (i / (width * height)) % out_channels;
    let batch = i / (width * height * out_channels);
    var total = 0.0;
    for (var input = 0u; input < in_channels; input++) {
        for (var ky = 0u; ky < kernel; ky++) {
            let iy = i32(y * stride + ky) - padding;
            if iy < 0 || iy >= i32(p.ints[2]) {
                continue;
            }
            for (var kx = 0u; kx < kernel; kx++) {
                let ix = i32(x * stride + kx) - padding;
                if ix < 0 || ix >= i32(p.ints[3]) {
                    continue;
                }

                let feature = (input * kernel + ky) * kernel + kx;
                total += a[input_index(batch, input, u32(iy), u32(ix))] * b[p.b_offset + feature * p.b_strides[0] + channel * p.b_strides[1]];
            }
        }
    }
    if p.flags != 0u {
        total += c[p.c_offset + channel * p.c_strides[0]];
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = total;
}

@compute @workgroup_size(64)
fn conv_input_gradients(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let in_channels = p.ints[1];
    let height = p.ints[2];
    let width = p.ints[3];
    let kernel = p.ints[7];
    let stride = i32(p.ints[8]);
    let x = i % width;
    let y = (i / width) % height;
    let input = (i / (width * height)) % in_channels;
    let batch = i / (width * height * in_channels);
    var total = 0.0;
    for (var channel = 0u; channel < p.ints[4]; channel++) {
        for (var ky = 0u; ky < kernel; ky++) {
            let padded_y = i32(y) + i32(p.ints[9]) - i32(ky);
            if padded_y < 0 || padded_y % stride != 0 {
                continue;
            }

            let oy = u32(padded_y / stride);
            if oy >= p.ints[5] {
                continue;
            }
            for (var kx = 0u; kx < kernel; kx++) {
                let padded_x = i32(x) + i32(p.ints[9]) - i32(kx);
                if padded_x < 0 || padded_x % stride != 0 {
                    continue;
                }

                let ox = u32(padded_x / stride);
                if ox >= p.ints[6] {
                    continue;
                }

                let feature = (input * kernel + ky) * kernel + kx;
                total += c[gradient_index(batch, channel, oy, ox)] * b[p.b_offset + feature * p.b_strides[0] + channel * p.b_strides[1]];
            }
        }
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = total;
}

@compute @workgroup_size(64)
fn conv_weight_gradients(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i >= p.count {
        return;
    }

    let kernel = p.ints[7];
    let channel = i % p.ints[4];
    let feature = i / p.ints[4];
    let kx = feature % kernel;
    let ky = (feature / kernel) % kernel;
    let input = feature / (kernel * kernel);
    var total = 0.0;
    for (var batch = 0u; batch < p.ints[0]; batch++) {
        for (var y = 0u; y < p.ints[5]; y++) {
            let iy = i32(y * p.ints[8] + ky) - i32(p.ints[9]);
            if iy < 0 || iy >= i32(p.ints[2]) {
                continue;
            }
            for (var x = 0u; x < p.ints[6]; x++) {
                let ix = i32(x * p.ints[8] + kx) - i32(p.ints[9]);
                if ix < 0 || ix >= i32(p.ints[3]) {
                    continue;
                }
                total += a[input_index(batch, input, u32(iy), u32(ix))] * c[gradient_index(batch, channel, y, x)];
            }
        }
    }

    dst[index_of(i, p.dst_strides, p.dst_offset)] = total;
}
