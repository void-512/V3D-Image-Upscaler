layout(set=0,binding=0,std430) readonly buffer In0 { float a[]; };
layout(set=0,binding=1,std430) readonly buffer In1 { float b[]; };
layout(set=0,binding=2,std430) buffer Out { float o[]; };
layout(set=0,binding=3,std430) readonly buffer Weights { float w[]; };
layout(set=0,binding=4,std430) readonly buffer Aux { float d[]; };
layout(set=0,binding=5,std430) readonly buffer Extra { float e[]; };
layout(push_constant,std430) uniform Params { int p[24]; };
// Abramowitz-Stegun erf (absolute error < 1.5e-7) for exact-form GELU.
float gelu(float x) {
    float z=abs(x)*0.7071067811865475;
    float t=1.0/(1.0+0.3275911*z);
    float r=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-z*z);
    return 0.5*x*(1.0+sign(x)*r);
}
float sigmoid(float x) { return 1.0/(1.0+exp(-x)); }
