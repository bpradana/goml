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
