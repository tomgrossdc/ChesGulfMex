/*
Notes on the ChesPartGL.cu

includes and global variable definitions
only ones that seem specific;
*/


// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;

int main(int argc, char **argv) {

runTest(argc,argv,ref_file);

}

bool runTest(int argc, char **argv, char *ref_file){

Sets up some initGL and glut callbacks for mouse etc.
registers the subroutine display() with glutDisplayFunc()
Then it creates the VBO 

// create VBO  Vertex Buffer Object
createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

This creates vbo, binds it to GL_ARRAY_BUFFER
allocates space to GL_ARRAY_BUFFER 
and seems to create vbo_res
cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags)

Back to runTest next call is 
runCuda(&cuda_vbo_resource);
glutMainLoop();   Spins on the routine display()
display() also keeps calling runCuda() so the previous call is a one time initializer

void runCuda(struct cudaGraphicsResource **vbo_resource){

    seems to use cudaGraphicsMapResources and
    cudaGraphicsResourceGetMappedPointer to update and 
    create dptr  a pointer to data being plotted

    Launches the cuda kernel  launch_kernel  which simply calls
    simple_vbo_kernel<<< grid, block>>>(pos, mesh_width, mesh_height, time);
the kernel is launched for each particle. Mapping using the grid,block
Seems inefficient, but whatever
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
Nothing complicated here.  Just creates new point pos[]
using time variable which was passed from runCuda as global g_fAnim
g_fAnim += 0.01f;   //inside display() 
}

display() after runCuda 
resets the display rotations translations 
//render from the vbo  
    glColor3f(1.0, 0.0, 0.0);   // sets point color


So coordinates of P[] are assigned in simple_vbo_kernel
Can that be replaced with a move kernel function? 
Need to embed move and the data reader into display()
Need to pass to display PPart MMesh and DData structures
Copy PPart to pos[ip] = make_float4(x_present[ip],z_present[ip],y_present[ip],1.0f);

Might be easier to add the openGL stuff to Cpp_Particle
Still have to construct overall makefile



// Trying to figure out the makefile
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      
-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 
-gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 
-gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 
-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 
-gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 
-o ChesPartGL ChesPartGL.o  
-L/usr/lib/nvidia-compute-utils-418 -lGL -lGLU -lglut

OK, now just include ChesPartGL.h and ChesPartGL.cu into mainpart.cu and 
no longer bother with compilation of ChesPartGL.o

particles version of makefile is only needed call.
make -f MakefileGL particles
   If you edit ChesPartGL.cu you have to touch mainpart.cu before make

Seems to work.  Now to figure out how to get PP into ChesPartGL.cu routines


Clean out the command line argc argv from the subroutines
mainChesPartGL(argc,argv)

    runTest(argc, argv, ref_file);
             int devID = findCudaDevice(argc, (const char **)argv);
             if (false == initGL(&argc, argv))


findCudaDevice is an nvidia helper function replace with:
devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors(cudaSetDevice(devID));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

initGL()
web recommends dummy argc, argv  , but try NULL NULL first
int argc = 1;
  char *argv[1] = {(char*)"Something"};
  glutInit(&argc, argv);

  Now to pass something to the graphic cuda

  Passing is working, one step at time. 
  mainChesPartGL (PP,MM,DD) > runTest 
  > runCuda > launch_kernel 
  >>glutMainLoop > display > runCuda



UNUSED CODE from display
   g_time_now tracks with dev_P[10].time_now  via +DT_SEC*CUDA_STEPS
Calculate time_frac
test if it is greater than 0.75
if so then increment the DD3 = [ 0,1,2,3] > [1,2,3,0]  or [1,2,3,0] > [ 2,3,0,1] etc.
     increment each DD3[i]=(DD3[i]+1)%4;
  And read in a new                          0                      1    etc.
    copy DD to dev_DD


//float time_frac=(g_time_now - DD[DD3[0]].time)/(DD[DD3[2]].time - DD[DD3[0]].time);
size_t DDSizeGeneral = sizeof(DData)*4;
cudaMemcpy(DD, dev_DD,DDSizeGeneral,cudaMemcpyDeviceToHost);
float time_frac=(g_time_now - DD[0].time); //(DD[2].time - DD[0].time);
 if (time_frac>.75){
    for (int i=0; i<4 ; i++)DD3[i]=(DD3[i]+1)%4;
    // what will follow DD3[2] well DD3[3] of course ? 
    //ReadData(g_time_now, DD3[3], DD, MM);
    size_t DDSizeGeneral = sizeof(DD);
    //cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);
}


Managed to put some stuff into runCuda  (which is called by display)
More I look at it, runCuda is not really necessary.  Could stuff that code 
into display direct.  runCuda does not solve the invisibility of MM, DD.
Did this:

   
    move<<< 256,64 >>>(dptr,dev_P,dev_MM,dev_DD,g_time_now);
    g_time_now += CUDA_STEPS* DT_SEC;   // 0.01f;
    
    if (int(g_time_now)%3600 < int((CUDA_STEPS*DT_SEC)/2.)){
        printf(" hourly g_time_now %g  DD3[0]=%d\n",g_time_now,DD3[0]);
        size_t DDSizeGeneral = sizeof(DData)*4;
        DData *DD;
        DD =  (DData *)malloc(4*sizeof(DData));
        cudaMemcpy(DD, dev_DD,DDSizeGeneral,cudaMemcpyDeviceToHost);
        MMesh *MM;
        MM = (MMesh *)malloc(sizeof(MMesh));
        cudaMemcpy(MM, dev_MM,sizeof(MMesh),cudaMemcpyDeviceToHost);


        printf(" DD[0].time = %g  DD3[3]=%d \n",DD[0].time,DD3[3]);
        DD[0].time+=3600.;
        DD[1].time+=3600.;
        DD[2].time+=3600.;
        ReadData(g_time_now, DD3[3], DD, MM);
        for (int i=0; i<3;i++){
            DD[i].time+=3600.;
            float time_nowrd=DD[i].time;
            ReadData(time_nowrd, i, DD, MM);

        }
        cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);
        
    

Oct 30, 2019 
things are going well. 
Now to add a thread for the data read
Outline:
Initialize with call to ReadData  three times
move to cuda mem
Thread last call to ReadData

Start main loop using the first three DD
On time_frac>.75 
  sync thread close it,  probably done anyway. 
  move to cuda mem 
  update the DD3 array 
  thread call to ReadData 
 Continue the loop
 
/*

Tasks,  Dec. 15, 2019
RunCuda:    
Move the read of a new hour data to in front of the move<<<>>> 
so it can thread while the particles are being move.  Still need to pass the thread  

Makefile:
Get rid of all the excess stuff.  Identify the NVIDIA required parts.

Data.cpp:
Read 3D data. Confirm that UVW's below Depth are set to zero.
Then make a 3D move.  
*/

 move3d  the new kernel

 /* //Backbone of new cuda move3d
// update position for the three meshes   iMM 012 UVW  
//           note the iMM=2 also gives ANGLE and depth
// Sets PP[Ip].i_ele4[iMM] and the factors  PP[Ip].factor4[iMM][0:2] 
for (in iMM=0; iMM<3; iMM++)
findiele(Ip,iMM,PP,MM);   

// find 2d interpolated ANGLE(icase==0), depth(icase==1)
//  input is X,Y, A of i_ele points along with factor4
MM[iMM].X[i_ele4[0:2]] MM[iMM].Y[i_ele4[0:2]] MM[iMM].ANGLE[i_ele4[0:2]] 
PP[Ip].factor4[iMM][i_ele[0:]]    
interpolate2D(Ip,iMM,PP,MM,icase);  // icase = 0U, 1V, 2W, 3ANGLE, 4depth  
maybe do  VAR[3] = MM[iMM].ANGLE[i_ele4[0:2]]  instead of icase 
That way we can feed it the vertical interpolates of UVW[3]
angle = 2Dinterpolate(Ip,iMM,PP,MM,3);    
depth = 2Dinterpolate(Ip,iMM,PP,MM,4); 

// vertical interpolates of UVW at three points
// for(i=0:2) iele=i_ele[i]; 
// for iz=0:nsigma find izp, izm 
// fact=z_present(izp-izm) 
// U[i]=PPU[izp]*fc +PPU[izm]*(1-fc)
//      do a DDT loop to find the three time steps for U's
U[3] = Vertinterpolate(depth,Ip,iMM,PP,MM,DD, icase=0);
V[3] = Vertinterpolate(depth,Ip,iMM,PP,MM,DD, icase=1);
W[3] = Vertinterpolate(depth,Ip,iMM,PP,MM,DD, icase=2);

//  Apply the factor4 's to U[3] to get U,V,W
//  Apply angle to U,V
//  Time average the three time steps of the UVW's
//  Time step PP.X,Y,Z  

*/

