var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn matmul_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = id.y;
    let column = id.x;
    let m = p.ints[0];
    let k = p.ints[1];
    let n = p.ints[2];
    let slot = local.y * 16u + local.x;
    var total = 0.0;
    for (var start = 0u; start < k; start += 16u) {
        tile_a[slot] = 0.0;
        tile_b[slot] = 0.0;
        if row < m && start + local.x < k {
            tile_a[slot] = a[p.a_offset + row * p.a_strides[0] + (start + local.x) * p.a_strides[1]];
        }
        if column < n && start + local.y < k {
            tile_b[slot] = b[p.b_offset + (start + local.y) * p.b_strides[0] + column * p.b_strides[1]];
        }

        workgroupBarrier();
        for (var inner = 0u; inner < 16u; inner++) {
            total += tile_a[local.y * 16u + inner] * tile_b[inner * 16u + local.x];
        }

        workgroupBarrier();
    }
    if row < m && column < n {
        dst[p.dst_offset + row * p.dst_strides[0] + column * p.dst_strides[1]] = total;
    }
}
