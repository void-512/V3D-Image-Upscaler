// Keep paired values in uint carriers across shuffles and reductions. The
// private V3DV patch recognizes the explicit typed vector operation.
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

f16vec2 half_values(uint word) {
    return uint16BitsToFloat16(u16vec2(word, word >> 16));
}
uint half_word(f16vec2 value) {
    u16vec2 bits = float16BitsToUint16(value);
    return uint(bits.x) | (uint(bits.y) << 16);
}
uint half_product(uint a, uint b) {
    precise f16vec2 product = half_values(a) * half_values(b);
    return half_word(product);
}
uint half_sum(uint a, uint b) {
    precise f16vec2 sum = half_values(a) + half_values(b);
    return half_word(sum);
}
uvec2 half_quad(vec4 value) {
    return uvec2(packHalf2x16(value.xy), packHalf2x16(value.zw));
}
vec4 float_quad(uvec2 words) {
    return vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
}
uint broadcast_low(uint word) {
    return (word & 65535u) | (word << 16);
}
uint broadcast_high(uint word) {
    return (word >> 16) | (word & 0xffff0000u);
}

#ifdef PACKED_PRODUCT_FP32
#define ACCUMULATOR vec4
vec4 update_row(vec4 sum, uint inputWord, uvec2 weights) {
    return sum + float_quad(uvec2(half_product(inputWord, weights.x),
                                  half_product(inputWord, weights.y)));
}
#else
#define ACCUMULATOR uvec2
uvec2 update_row(uvec2 sum, uint inputWord, uvec2 weights) {
    return uvec2(half_sum(sum.x, half_product(inputWord, weights.x)),
                 half_sum(sum.y, half_product(inputWord, weights.y)));
}
#endif

void update_tile(uvec2 a, uvec2 b, inout ACCUMULATOR s0, inout ACCUMULATOR s1,
                 inout ACCUMULATOR s2, inout ACCUMULATOR s3) {
    s0 = update_row(s0, broadcast_low(a.x), b);
    s1 = update_row(s1, broadcast_high(a.x), b);
    s2 = update_row(s2, broadcast_low(a.y), b);
    s3 = update_row(s3, broadcast_high(a.y), b);
}
