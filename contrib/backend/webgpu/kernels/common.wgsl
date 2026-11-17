struct Params {
    count: u32,
    rank: u32,
    op: u32,
    flags: u32,
    shape: array<u32, 8>,
    a_strides: array<u32, 8>,
    b_strides: array<u32, 8>,
    c_strides: array<u32, 8>,
    dst_strides: array<u32, 8>,
    a_offset: u32,
    b_offset: u32,
    c_offset: u32,
    dst_offset: u32,
    ints: array<u32, 24>,
    scalars: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read> p: Params;

fn linear_id(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.x + id.y * groups.x * 64u;
}

fn index_of(linear: u32, strides: array<u32, 8>, offset: u32) -> u32 {
    var rest = linear;
    var index = offset;
    for (var axis = i32(p.rank) - 1; axis >= 0; axis--) {
        index += (rest % p.shape[axis]) * strides[axis];
        rest /= p.shape[axis];
    }

    return index;
}

fn nan_value() -> f32 {
    return bitcast<f32>(0x7fc00000u);
}

fn infinity() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn is_nan(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}

fn is_finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        return 1.0 / (1.0 + exp(-value));
    }

    let exponential = exp(value);
    return exponential / (1.0 + exponential);
}

fn tanh_value(value: f32) -> f32 {
    if is_nan(value) || value == 0.0 {
        return value;
    }

    if !is_finite(value) {
        return select(-1.0, 1.0, value > 0.0);
    }

    return tanh(value);
}
