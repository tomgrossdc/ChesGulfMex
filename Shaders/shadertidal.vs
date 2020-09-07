#version 330
//shadertidal.vs

layout (location = 0) in vec4 Position;

uniform mat4 gWVP;

out vec4 Color;

void main()
{
    // pos[Ipx] = make_float4(scale*PP[Ip].x_present,-scale*PP[Ip].z_present,-scale*PP[Ip].y_present,  1.0f);

   float scale = 0.00002;  // reduce XY meters to +-1.0 
   float vertscale = .02;  // reduce Z meters to 0.0 -.2
 
    gl_Position = gWVP*vec4(scale*Position[0],vertscale*Position[2],-scale*Position[1],1.0);   // gWVP * vec4
    gl_Position[2] = - gl_Position[2];

    // orange to blue for depth
    vec4 rr=vec4(1.,    .7333, 0.   , .3176); //, .4471, .1961);
    vec4 gg=vec4(.8314, .8902, .6667, .7529); //, .6235, .3137);
    vec4 bb=vec4(.1569, .2392, .2   , .698 ); //, .8188, .6863);
    vec4 rrr=vec4( .3176, .4471, .1961, 0.0);
    vec4 ggg=vec4( .7529, .6235, .3137, 0.0);
    vec4 bbb=vec4( .698 , .8188, .6863, 0.99);
// Four colors for the range around depth +-.4m  +-depthcolorinterval/10
/*
    vec4 rs=vec4( .5, .25, .5, 0.0);
    vec4 gs=vec4( .0 , .0,  .25, 0.25);
    vec4 bs=vec4( .5, .0,  .00, 0.25);
*/
    vec4 rs=vec4( .99, .5, .9, 0.0);
    vec4 gs=vec4( .0 , .0,  .5, 0.95);
    vec4 bs=vec4( .99, .0,  .00, 0.95);



//  scalecolor = 0:6 after subtracting 2.
    float scalecolor = Position[3]-2.;
    float cs=scalecolor;
    int ic, icp;
    float r=0.; float g=0.; float b=0.;
    float h=.5;

    // 0-1.0 ranges +4m to 0m , invert it to do 0m to .4m
    if (scalecolor<1.0)
    {ic=0; icp=1; 
    cs =  1.-10.*(1.- scalecolor-0) ;
    r=rs[ic]+(cs)*(rs[icp]-rs[ic]);
    g=gs[ic]+(cs)*(gs[icp]-gs[ic]);
    b=bs[ic]+(cs)*(bs[icp]-bs[ic]); 
    //r=.25+(cs)*(1. -.25);
    //g= 0.+(cs)*(0. -0.);
    //b=.75+(cs)*(0. -.75); 
    h=1.;
    }
    // 1.0-2.0 ranges 0m to -4m , reduce it to do 0m to -.4m
    else if (scalecolor<1.3)
    {
        ic=2; icp=3; cs = (scalecolor-1.0)/.3;
        r=rs[ic]+(cs)*(rs[icp]-rs[ic]);
        g=gs[ic]+(cs)*(gs[icp]-gs[ic]);
        b=bs[ic]+(cs)*(bs[icp]-bs[ic]);
        h=1.; 
    }

    else if (scalecolor<2.0)
    {ic=0; icp=2; cs = (scalecolor-1.3)/.7;
    r=rr[ic]+(cs)*(rr[icp]-rr[ic]);
    g=gg[ic]+(cs)*(gg[icp]-gg[ic]);
    b=bb[ic]+(cs)*(bb[icp]-bb[ic]); }

    else if (scalecolor<3.0)
     {ic=2; icp=3; cs = scalecolor-2.0;
    r=rr[ic]+(cs)*(rr[icp]-rr[ic]);
    g=gg[ic]+(cs)*(gg[icp]-gg[ic]);
    b=bb[ic]+(cs)*(bb[icp]-bb[ic]); }

   else if (scalecolor< 4.0)
     {ic=0; icp=1; cs = scalecolor-3.0;
    r=rrr[ic]+(cs)*(rrr[icp]-rrr[ic]);
    g=ggg[ic]+(cs)*(ggg[icp]-ggg[ic]);
    b=bbb[ic]+(cs)*(bbb[icp]-bbb[ic]); }
    
    else if (scalecolor< 5.0)
     {ic=1; icp=2; cs = scalecolor-4.0;
    r=rrr[ic]+(cs)*(rrr[icp]-rrr[ic]);
    g=ggg[ic]+(cs)*(ggg[icp]-ggg[ic]);
    b=bbb[ic]+(cs)*(bbb[icp]-bbb[ic]); }

    else if (scalecolor<6.0)
    {ic=2; icp=3; cs = scalecolor-5.0;
    r=rrr[ic]+(cs)*(rrr[icp]-rrr[ic]);
    g=ggg[ic]+(cs)*(ggg[icp]-ggg[ic]);
    b=bbb[ic]+(cs)*(bbb[icp]-bbb[ic]); }
    else
   {r=rrr[3]; g = ggg[3]; b=bbb[3]; }

    Color = vec4(r,g,b,h);   // below -.05
    if (Position[3] < 1.001)
    {    Color = vec4(0.0,0.0,0.0,.01);   // grounded, nearly invisible
    }
    if (Position[3] < 0.0001) 
    {    Color = vec4(1.0,1.0,1.0,1.0);   // white border
    }

    //scalecolor = 1.0-scalecolor;
    //r=scalecolor/steps; g=1.-scalecolor/steps; b=0.0;

    //Color = vec4(r,g,b,scalecolor/steps);   // below -.05





}
