#include "packed_math.glsl"
layout(set=0,binding=6,std430) readonly buffer HalfInput {uvec2 pairedInput[];};
layout(set=0,binding=5,std430) readonly buffer HalfWeights {uvec2 pairedWeights[];};

void packed_gemm(uint ai, uint bi, uint bs, uint ar, uint br,
                 out vec4 s0, out vec4 s1, out vec4 s2, out vec4 s3) {
    s0=vec4(0); s1=vec4(0); s2=vec4(0); s3=vec4(0);
#ifdef PACKED_PRODUCT_FP32
    uint first=0u, end=uint(IC);
    {
#else
    for(uint first=0u;first<uint(IC);first+=16u){
        uint end=min(first+16u,uint(IC));
#endif
        ACCUMULATOR r0=ACCUMULATOR(0),r1=ACCUMULATOR(0);
        ACCUMULATOR r2=ACCUMULATOR(0),r3=ACCUMULATOR(0);
        for(uint k=first;k<end;k+=4u){
            uvec2 av=pairedInput[ai+k],bv=pairedWeights[bi+k/4u*bs];
            update_tile(av,bv,r0,r1,r2,r3);
            av=subgroupShuffle(av,ar);bv=subgroupShuffle(bv,br);
            update_tile(av,bv,r0,r1,r2,r3);
            av=subgroupShuffle(av,ar);bv=subgroupShuffle(bv,br);
            update_tile(av,bv,r0,r1,r2,r3);
            av=subgroupShuffle(av,ar);bv=subgroupShuffle(bv,br);
            update_tile(av,bv,r0,r1,r2,r3);
        }
#ifdef PACKED_PRODUCT_FP32
        s0=r0;s1=r1;s2=r2;s3=r3;
#else
        s0+=float_quad(r0);s1+=float_quad(r1);
        s2+=float_quad(r2);s3+=float_quad(r3);
#endif
    }
}
