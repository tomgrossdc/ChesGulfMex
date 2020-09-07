#version 330
//shaderpipe.vs

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

    /*
    // orange to blue for depth
    vec4 rr=vec4(1.,    .7333, 0.   , .3176); //, .4471, .1961);
    vec4 gg=vec4(.8314, .8902, .6667, .7529); //, .6235, .3137);
    vec4 bb=vec4(.1569, .2392, .2   , .698 ); //, .8188, .6863);
    vec4 rrr=vec4( .3176, .4471, .1961, 0.0);
    vec4 ggg=vec4( .7529, .6235, .3137, 0.0);
    vec4 bbb=vec4( .698 , .8188, .6863, 0.99);
*/
    // rainbow colors for pipe releases
    vec4 rr= vec4(1.,1.,0.,0.);
    vec4 gg= vec4(0.,1.,1.,1.);
    vec4 bb= vec4(0.,0.,0.,1.);
    vec4 rrr=vec4(0.,0.,0.,1. );
    vec4 ggg=vec4(1.,0.,0.,0. );
    vec4 bbb=vec4(1.,1.,1.,0. );


//  scalecolor = 0:6 after subtracting 2.
    float scalecolor = Position[3]-2.;
    float cs=scalecolor;
    int ic, icp;
    float r=0.; float g=0.; float b=0.;

    if (scalecolor<1.0)
    {ic=0; icp=1; cs = scalecolor-0.0;
    r=rr[ic]+(cs)*(rr[icp]-rr[ic]);
    g=gg[ic]+(cs)*(gg[icp]-gg[ic]);
    b=bb[ic]+(cs)*(bb[icp]-bb[ic]); }

    else if (scalecolor<2.0)
    {ic=1; icp=2; cs = scalecolor-1.0;
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

    Color = vec4(r,g,b,1.0);   // below -.05
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
