// Keep pairs in uint carriers across shuffles and use bitcasts for FP16 math.
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

uint half_product(f16vec2 a, uint b) {
    precise f16vec2 product = a * unpackFloat2x16(b);
    return packFloat2x16(product);
}
uint half_sum(uint a, uint b) {
    precise f16vec2 sum = unpackFloat2x16(a) + unpackFloat2x16(b);
    return packFloat2x16(sum);
}
uvec2 half_quad(vec4 value) {
    return uvec2(packHalf2x16(value.xy), packHalf2x16(value.zw));
}
vec4 float_quad(uvec2 words) {
    return vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
}
#ifdef PACKED_PRODUCT_FP32
#define ACCUMULATOR vec4
vec4 update_row(vec4 sum, f16vec2 pair, uvec2 weights) {
    return sum + float_quad(uvec2(half_product(pair, weights.x),
                                  half_product(pair, weights.y)));
}
#else
#define ACCUMULATOR uvec2
uvec2 update_row(uvec2 sum, f16vec2 pair, uvec2 weights) {
    return uvec2(half_sum(sum.x, half_product(pair, weights.x)),
                 half_sum(sum.y, half_product(pair, weights.y)));
}
#endif

void update_tile(uvec2 a, uvec2 b, inout ACCUMULATOR s0, inout ACCUMULATOR s1,
                 inout ACCUMULATOR s2, inout ACCUMULATOR s3) {
    f16vec2 lo = unpackFloat2x16(a.x), hi = unpackFloat2x16(a.y);
    s0 = update_row(s0, lo.xx, b);
    s1 = update_row(s1, lo.yy, b);
    s2 = update_row(s2, hi.xx, b);
    s3 = update_row(s3, hi.yy, b);
}
