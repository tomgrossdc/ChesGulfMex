#version 330

layout (location = 0) in vec4 Position;

uniform mat4 gWVP;

out vec4 Color;

void main()
{
    // pos[Ipx] = make_float4(scale*PP[Ip].x_present,-scale*PP[Ip].z_present,-scale*PP[Ip].y_present,  1.0f);

   float scale = 3.25;
    //gl_Position = vec4(Position, 1.0);   // gWVP * vec4
    //gl_Position = vec4(scale*Position[0],-scale*Position[1],scale*Position[2],1.0);   // gWVP * vec4
    //gl_Position = gWVP*vec4(scale*Position[0],.5*scale*Position[2],-scale*Position[1],1.0);   // gWVP * vec4
    gl_Position = gWVP*vec4(scale*Position[0],0.75*scale*Position[2],-scale*Position[1],1.0);   // gWVP * vec4
    //Color = vec4(clamp(Position, 0.0, 1.0), 1.0);
    float steps=6.;    // Make scalecolor go from 0 to steps instead of just 0 to 1.0
    float scalecolor = steps-(Position[3]/(1.*10600.));     //  57600.
    while (scalecolor>steps ) scalecolor=scalecolor -steps;
    while (scalecolor<0.0 ) scalecolor=scalecolor +steps;
    float r=0.; float g=0.; float b=0.;
    if (scalecolor<1.0)
    {g=(scalecolor    )*steps; r=1.0; b=0.0; }
    else if (scalecolor<2.0)
    {r=1.-(scalecolor-1.0    ); g=1.0; b=0.; }
    else if (scalecolor<3.0)
    {b=(scalecolor-2.0    ); g=1.0; r=0.; }
    else if (scalecolor<4.0)
    {g=1.-(scalecolor-3.0    ); b=1.0; r=0.; }
    else if (scalecolor< 5.0)
    {r=(scalecolor-4.0    ); b=1.0; g=0.; }
    else
    //{g=(scalecolor    )*6.; r=1.0; b=0.0; }
    {b=1.-(scalecolor-5.0); r=1.0; g=0.; } 

    if (Position[3] < .1) 
    {r=1.; g=1.; b=1.;}

    //scalecolor = 1.0-scalecolor;
    //r=scalecolor/steps; g=1.-scalecolor/steps; b=0.0;

    Color = vec4(r,g,b,.5);   // below -.05
    //Color = vec4(r,g,b,scalecolor/steps);   // below -.05





}
