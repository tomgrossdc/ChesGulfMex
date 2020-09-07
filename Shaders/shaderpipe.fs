#version 330
// shaderpipe.fs

in vec4 Color;

out vec4 FragColor;

void main()
{
    FragColor = Color;

    //float depth = gl_FragCoord.z ;
    //FragColor = vec4(vec3(depth),1.0);
}
