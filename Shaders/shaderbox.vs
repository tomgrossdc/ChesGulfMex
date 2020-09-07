#version 330
//shaderfill.vs

layout (location = 0) in vec4 Position;

uniform mat4 gWVP;

out vec4 Color;

void main()
{
    // pos[Ipx] = make_float4(scale*PP[Ip].x_present,-scale*PP[Ip].z_present,-scale*PP[Ip].y_present,  1.0f);

   float scale = 0.00002;  // reduce XY meters to +-1.0 
   float vertscale = .02;  // reduce Z meters to 0.0 -.2
    //gl_Position = vec4(Position, 1.0);   // gWVP * vec4
    //gl_Position = vec4(scale*Position[0],-scale*Position[1],scale*Position[2],1.0);   // gWVP * vec4
    //gl_Position = gWVP*vec4(Position[0],Position[1],Position[2],1.0);   // gWVP * vec4
    //Color = vec4(clamp(Position, 0.0, 1.0), 1.0);
    gl_Position = gWVP*vec4(scale*Position[0],vertscale*Position[2],-scale*Position[1],1.0);   // gWVP * vec4

    //if (Position[2]>0.001)
    //      Color = vec4(1.,1.,0., 1.0);  // above surface
    //else 
    Color = vec4(1.0,0.75,0.75,1.0);    // default color for dots above sea level
    if (Position[2]<0.05){   // exclude dots above sea level
    if (Position[2]>-5.)                 // Z no longer multiplied by 1000*scale,  use real meters
      Color = vec4(1.,0.,0., 0.50);  // surface above -2.5 m
    else if (Position[2]>-10.)
      Color = vec4(0.,1.,0.,0.50);   // between -2.5 -5.
    else
      Color = vec4(0.,0.,1.0,0.50);   // below -5.0
    }
    if (Position[3] < .1) 
      Color = vec4(1.,1.,1.0,1.0);   // border points
}
