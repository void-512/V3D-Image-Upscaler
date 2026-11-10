#version 450
void main() {
    vec2 xy=vec2((gl_VertexIndex<<1)&2,gl_VertexIndex&2);
    gl_Position=vec4(xy*2.-1.,0.,1.);
}
