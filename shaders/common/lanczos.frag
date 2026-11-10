#version 450
// Pixel-center Lanczos3, widened by the decimation ratio. Coefficients below
// are precomputed sinc(t)*sinc(t/3), normalized in double precision, rounded
// once to FP32. Symmetric pairs halve FMAs; no hardware linear filtering.
layout(constant_id=0) const int sh=128;
layout(constant_id=1) const int sw=128;
layout(constant_id=2) const int ratio=2;
layout(constant_id=3) const int anime=0;
layout(constant_id=4) const int textureWidth=64;
layout(constant_id=5) const int rasterWidth=64;
#if PASS == 1
layout(set=0,binding=0,std430) readonly buffer Input { float a[]; };
layout(location=0) out vec4 horizontal;
#else
layout(set=0,binding=1) uniform sampler2D intermediate;
layout(set=0,binding=2,std430) writeonly buffer Output { float o[]; };
#endif
vec3 fetchValue(int x,int y) {
#if PASS == 1
    if(anime==0){int j=(y*sw+x)*3;return vec3(a[j],a[j+1],a[j+2]);}
    int j=((y/4)*(sw/4)+x/4)*48+(y%4)*4+x%4;
    return vec3(a[j],a[j+16],a[j+32]);
#else
    if(textureWidth==sw/ratio)return texelFetch(intermediate,ivec2(x,y),0).rgb;
    int j=y*(sw/ratio)+x;
    return texelFetch(intermediate,ivec2(j%textureWidth,j/textureWidth),0).rgb;
#endif
}
vec3 pairValue(int k,int x,int y) {
#if PASS == 1
    int q=x*ratio-5*ratio/2;
    return fetchValue(clamp(q+k,0,sw-1),y)+fetchValue(clamp(q+6*ratio-1-k,0,sw-1),y);
#else
    int q=y*ratio-5*ratio/2;
    return fetchValue(x,clamp(q+k,0,sh-1))+fetchValue(x,clamp(q+6*ratio-1-k,0,sh-1));
#endif
}
void main() {
    int width=sw/ratio;
#if PASS == 1
    int height=sh;
#else
    int height=sh/ratio;
#endif
    ivec2 xy=ivec2(gl_FragCoord.xy);
    int i=xy.y*rasterWidth+xy.x;
    if(i>=width*height)discard;
    if(rasterWidth!=width)xy=ivec2(i%width,i/width);
    int x=xy.x,y=xy.y;
    vec3 sum=vec3(0.);
if(ratio==2){
sum=fma(pairValue(0,x,y),vec3(0.0036891354301046677),sum);
sum=fma(pairValue(1,x,y),vec3(0.01505614268094883),sum);
sum=fma(pairValue(2,x,y),vec3(-0.033998631514275908),sum);
sum=fma(pairValue(3,x,y),vec3(-0.06663731776798075),sum);
sum=fma(pairValue(4,x,y),vec3(0.13550528412853946),sum);
sum=fma(pairValue(5,x,y),vec3(0.44638538704266367),sum);
}else{
sum=fma(pairValue(0,x,y),vec3(0.00046058079093877475),sum);
sum=fma(pairValue(1,x,y),vec3(0.0039105746530807431),sum);
sum=fma(pairValue(2,x,y),vec3(0.0075994022974713157),sum);
sum=fma(pairValue(3,x,y),vec3(0.005124277451910025),sum);
sum=fma(pairValue(4,x,y),vec3(-0.0076647263200382695),sum);
sum=fma(pairValue(5,x,y),vec3(-0.026437560067430477),sum);
sum=fma(pairValue(6,x,y),vec3(-0.036925187201617776),sum);
sum=fma(pairValue(7,x,y),vec3(-0.021290906444550704),sum);
sum=fma(pairValue(8,x,y),vec3(0.030222779257183623),sum);
sum=fma(pairValue(9,x,y),vec3(0.10973536917548576),sum);
sum=fma(pairValue(10,x,y),vec3(0.19161815800095638),sum);
sum=fma(pairValue(11,x,y),vec3(0.24364723840661062),sum);
}
#if PASS == 1
    horizontal=vec4(sum,0.);
#else
    if(anime!=0)sum+=vec3(.4488,.4371,.4040);
    o[i*3]=sum.r;o[i*3+1]=sum.g;o[i*3+2]=sum.b;
#endif
}
