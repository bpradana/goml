@compute @workgroup_size(64)
fn copy_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i < p.count {
        dst[index_of(i, p.dst_strides, p.dst_offset)] = a[index_of(i, p.a_strides, p.a_offset)];
    }
}

@compute @workgroup_size(64)
fn fill_values(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(id, groups);
    if i < p.count {
        dst[index_of(i, p.dst_strides, p.dst_offset)] = p.scalars.x;
    }
}

@compute @workgroup_size(64)
fn finite_values(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }

    var invalid = 0.0;
    for (var i = 0u; i < p.count; i++) {
        if !is_finite(a[index_of(i, p.a_strides, p.a_offset)]) {
            invalid = 1.0;
            break;
        }
    }

    dst[0] = invalid;
}
