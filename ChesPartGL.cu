////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

/*
ChesPartGL.h and ChesPartGL.cu are included at top of mainpart.cu
#include "ChesPartGL.h"
*/

#include <typeinfo>

// MakeShader.cpp
bool ReadFileNew(string pFileName, string& outFile);
static void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType);
void CompileShaders(string vs, string fs);
void transmatrix(float *matrix, float tx,float ty, float tz);


// flag for pausing motion in cudamove
bool runaction=true;
bool getRunAction(){ return runaction; }

// thread method to be developed for disk reading
//std::thread t12 ;
void UpDData();
int DDataNum = 10;

curandState_t* states;

void UpDData()
{
    // Thread programme to run continuously in background 
    // Will change value of a global counter

    while(DDataNum>=0)
    {
        if (DDataNum==0) {
            // DDataNum == 0 is flag to cause a full datafile read to DD[0].DD3[3]
            printf("\n\n ******************Read DDataFile to DD3[3] = %d  ",DD[0].DD3[3]);

            string newername;
            if (DD[0].IsFVCOM) 
              {
                newername = NetCDFfiledateG(DD[0].filetemplate,DD);
                ReadFieldNetCDFG(newername,DD[0].DD3[3],DD,MM);
              }
            else
              {
                newername = NetCDFfiledate(DD[0].filetemplate,DD);
                ReadFieldNetCDF(newername,DD[0].DD3[3],DD,MM);
            }
            printf("***************** Finished Read DDataFile \n");
            cout<< newername << endl<<endl;
        
        }

    
        DDataNum+=1;
        //printf(" display i=%d, DDataNum=%d \n",i, DDataNum);
        this_thread::sleep_for(chrono::milliseconds(2) );
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    //char fps[256];
    //sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    //glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL()
{
    int argc = 1;
    char *argv[1] = {(char*)"Something"};
    glutInit(&argc, argv);
    //glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Regular NetCDF Particle Tracking");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    // background color  234,255,199,1  is noaa map color 
    //glClearColor(234./256,225./256.,199./256.,0.0);  // NOAA color  
    glClearColor(234./256.-.2,225./256.-.2,199./256.-.2,0.0);  // Darker NOAA color 
    //glClearColor(0.,0.,0.,1.0);  // black or grey: 0.0050, 0.005, 0.0050, 1.0);  
    glColor4f(0.0,1.0,0.0,1.0);   // set color
    //glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 20.0); 
    //  near and far clipping planes  .1,10.  

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//  Initialize a few gl, timer and cuda 
//  then start gl looping to call function display
////////////////////////////////////////////////////////////////////////////////
bool GLmoveparticle(struct PPart *PP, struct MMesh *MM, struct DData *DD) 
{
    //, struct CCcontrol *CC)
    //int DD3[4];
    // Create the CUTIL timer
    sdkCreateTimer(&timer);
    g_time_now = (DD[0].time + DD[1].time)/2.0; 
    Dot_Size = MM[0].Dot_Size;
    
    //initial the cudaDevice to use, as if there is a choice?
    cudaDeviceProp deviceProp;
    int devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n"
         , devID, deviceProp.name, deviceProp.major, deviceProp.minor);


        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        //if (false == initGL(&argc, argv))        
        if (false == initGL())

        {
            return false;
        }

        // register callbacks.   these are locally defined functions
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

    // create VBO
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

    // routine in MakeShader.cpp Creates gWVPLocation for use by RenderSceneCB()
    CompileShaders(MM[0].shadervs, MM[0].shaderfs);    



        // Launch UpDData() threaded.  Will run in background till end of time
        std::thread t(UpDData );
        //std::thread t1g(ReadFieldNetCDFG, std::ref(newername),std::ref(DD[0].DD3[3]),
        //                std::ref(DD),std::ref(MM) );
                

        // run the cuda part from routine display 
        // specified in glutDisplayFunc(display);
        // which is triggered by glutMainLoop
        //runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        printf(" Start glutMainLoop  >display>runCuda \n\n");

        glutMainLoop();

        printf(" Return from glutMainLoop\n");

//    }

    return true;
}


/* this GPU kernel function is used to initialize the random states */
__global__ void initcurand(unsigned int seed, curandState_t* states) {
    int cudaindex = threadIdx.x + blockIdx.x * blockDim.x;

    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                cudaindex, /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[cudaindex]);
  }
////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation,  called from display
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    //printf("TFG runCuda host_P[10].x_present %g\n",host_P[10].x_present);
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    //float4 *dptr2;

    float time_now;

    size_t DDSizeGeneral = sizeof(DData)*4;
    size_t MMSizeGeneral = sizeof(MMesh)*4;

    if (iDD==-1){   // First update, need to localize DD, MM only once
                    // initialized in ChesPartGL.h, global to this file
                    
        printf("\n runCuda First Pass\n");
            try {
                printf(" Can I print DD[0].time_now %g\n",DD[0].time_now);
            } catch (const std::runtime_error& e){
                printf(" Error on print DD[0].time_now Message: %s\n",e.what());
            }
        cudaMemcpy(DD, dev_DD,DDSizeGeneral,cudaMemcpyDeviceToHost);
        cudaMemcpy(MM, dev_MM,MMSizeGeneral,cudaMemcpyDeviceToHost);
        printf(" After cudaMemcpy  DD[0].time_now %fsec %f hr\n",DD[0].time_now,DD[0].time_now/3600.);

        iDD=0;

        // Initialize the damn random number generator
        //  outside curandState_t* states;
        /* allocate space on the GPU for the random states */
        cudaMalloc((void**) &states, 256*64* sizeof(curandState_t));
        initcurand<<<256,64>>>(16343, states);

    }
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));    //1
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
    *vbo_resource));                                                                  // *vbo_resource
    //checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr2, &num_bytes,
    //vbo_resource[1]));
   

    //int DD33=1;   256,64  72*32 = 2304 cuda cores
    // 144,32  all cudas enumerated once
    move3d<<< 144,32 >>>(dptr,dev_P,dev_MM,dev_DD,states);
    cudaDeviceSynchronize();
    DD[0].time_now += CUDA_STEPS* DT_SEC;   // 0.01f;   
    time_now = DD[0].time_now;
//printf("After cuda move3d time_now = %fs %ghr\n",time_now, time_now/3600.);
    float time_frac=(time_now - DD[DD[0].DD3[0]].time)/(DD[DD[0].DD3[2]].time - DD[DD[0].DD3[0]].time);
    bool timetest =  (time_frac > .75);

    //  Dummy counter reset of UpDData flag
    //if (DDataNum>1000) { 
    //    printf("\n\n\n ****************\n RunCuda reset of DDataNum=%d\n ****************\n\n\n",DDataNum);
    //    DDataNum=0;
    //}



    if (timetest ){

        
        //  Every hour a new data file is needed. Read dev_DD to obtain time_now
        
        // Assume or test that the fourth ReadData thread is finished and move to dev_DD
        cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);
        
        //  Update DD3  
        for (int i=0; i<4 ; i++)DD[0].DD3[i]=(DD[0].DD3[i]+1)%4;

        
        // DD3[3] is next spot to be updated, will be updated in this section
        //  Thread this off to execute while elsewhere.
        //        printf(" DD[# 1].time = %g %g %g %g\n",DD[0].time/3600.,DD[1].time/3600.,DD[2].time/3600.,DD[3].time/3600.);
        
        DD[0].ToDay +=3600;  // for hourly files
        string newername;

/*
        if (DD[0].IsFVCOM) 
        {newername = NetCDFfiledateG(DD[0].filetemplate,DD);}
        else
        {newername = NetCDFfiledate(DD[0].filetemplate,DD);}
        //string newername = NetCDFfiledate(DD[0].filetemplate,DD);
        cout<< newername << endl;
        
        //char fps[256];
        //strftime(fps,80, "Chesapeake Bay  %A %G %b %d %r  ", gmtime(&DD[0].ToDay));
        //strftime(&fps[80],80, "more Time= %F %R.", gmtime(&MM[0].ToDay));
        //glutSetWindowTitle(fps);
        
        
        bool RunThreadRead = false;
        if (RunThreadRead)
        {
            if (DD[0].IsFVCOM) 
            {std::thread t1g(ReadFieldNetCDFG, std::ref(newername),std::ref(DD[0].DD3[3]),
                std::ref(DD),std::ref(MM) );
                t1g.join();   // Wait here for thread to finish. Makes threading moot.  Testing only.
            }
            else
            {
                //std::thread & t1;
                printf(" thread start t1(ReadFieldNetCDF  \n");
                std::thread t1(ReadFieldNetCDF, std::ref(newername),std::ref(DD[0].DD3[3]),
                std::ref(DD),std::ref(MM) );
                //printf(" thread after t1(ReadFieldNetCDF  \n");
                t1.join();   // Wait here for thread to finish. Makes threading moot.  Testing only.
                //t1.detach();    // Let it loose, but with no test for finished crashes
                //printf(" thread after t1.join()  \n");
                //std::thread t2(ReadFieldNetCDF, std::ref(newername),std::ref(DD[0].DD3[3]),
                //std::ref(DD),std::ref(MM) );
                //t2.join();
                //printf("after second join\n");
            }
            
        }
        else
        {
            if (DD[0].IsFVCOM) 
            {
                ReadFieldNetCDFG(newername,DD[0].DD3[3],DD,MM);
            }
            else
            {
                ReadFieldNetCDF(newername,DD[0].DD3[3],DD,MM);
            }
        }
*/        
        //  Reset of UpDData flag to cause call to read data by threaded UpDData()
        DDataNum=0;
        printf("\n\n\n ****************\n RunCuda DDataNum=%d\n ****************\n\n\n",DDataNum);
        



        float dhr=3600.;
        printf(" DD[     0:3].time = %g %g %g %g\n",DD[0].time/dhr,DD[1].time/dhr,DD[2].time/dhr,DD[3].time/dhr);
        
        iDD+=1;
        printf(" iDD = %d       time_now=%g\n\n",iDD,time_now/dhr);
    }    // End of hourly DD update

    char fps[256];
    time_t tnow;
    tnow =DD[0].ToDay;
    tnow = MM[0].Time_Init + time_now -MM[0].time_init; 
    //strftime(fps,80, "Chesapeake Bay3  %A %G %b %d %I:%M %R  ", gmtime(&DD[0].ToDay));
    strftime(fps,80, "Chesapeake Bay2  %A %G %b %d   %R ", gmtime(&tnow));
    //strftime(&fps[80],80, "more Time= %F %R.", gmtime(&DD[0].ToDay));
    glutSetWindowTitle(fps);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object 
    glGenBuffers(2, vbo);

    ////////////////////buffer number [0]
    //glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

    // initialize buffer object
    //unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    unsigned size = MAX_GLPARTICLES *4*sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    //checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));


    /* */
    ////////////////////buffer number [1]
   //glBindBuffer(GL_ARRAY_BUFFER, *vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

   // initialize buffer object
   //size = mesh_width * mesh_height * 4 * sizeof(float);
   size = MAX_GLPARTICLES *4*sizeof(float);
   glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

   //glBindBuffer(GL_ARRAY_BUFFER, 0);

   // register this buffer object with CUDA
   checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
/* */

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    if (getRunAction())
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    //glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    GLfloat idmatrix[16] = {1.0,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.};  //Identity

    float *newmatrix;
    newmatrix = matrix_RotTrPer(idmatrix, 
        rotate_y,rotate_x,0.0,
        translate_x,translate_y,translate_z,
        (GLfloat)window_width, (GLfloat) window_height, znear, zfar, 30.0);

    // gWVPLocation points to the variable "gWVP" in the shader.vs as 4x4matrix
    glUniformMatrix4fv(gWVPLocation, 1, GL_FALSE, newmatrix);
        
        
    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
   
    glPointSize(Dot_Size);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);   //  default was GL_LESS which gave backwards
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDrawArrays(GL_POINTS, 0, MAX_GLPARTICLES);

    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();    

}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
    
    

}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        case (112) :  // p pause
            {   
              // Set flag to stop cuda move, but keep refreshing screen for mousing
              //  p  pause action  toggles runaction
                if (runaction){ runaction=false;}
                else { runaction=true;}
                cout <<" Key = "<<key << " runaction "<<runaction<< endl;
            }    
            break ;
        case (104) :   // h help menu 
            {printf("\nesc = stop\n p = toggle pause\n h = this help\n j = narrow view plane \n k = expand view plane\n");
             printf(" Move: w^ s. a<  d> \n");} 
            break;
        case (106) :   // j reset view   Set matrix in display(). Doesn't really work too good. oh well.
            {
                znear = 1.0;
                zfar = 15.0;
                //printf(" j narrow view plane %g %g\n",znear, zfar);
                //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 1.0, 2.); 
                //  near and far clipping planes  .1,10.  or .1,20.
            }
            break;
        case (107) :   // k znear contract view 
            {
                znear += 0.1;
             //   zfar -= 1.0;
                //printf(" k contract znear view plane %g %g\n",znear, zfar);
                //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, .1, 20.); 

            }
            break;
        case (108) :   // l zfar contract view 
            {
              //  znear += 0.1;
                zfar -= 1.0;
                //printf(" l contract zfar view plane %g %g\n",znear, zfar);
                //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, .1, 20.); 

            }
            break;
        case(119) :  // w translate up
        {        translate_y += 0.03f;
        }  break;
        case(115) :  // s translate down
        {        translate_y -= 0.03f;
        }  break;
        case(97) :  // a translate left
        {        translate_x -= 0.03f;
        }  break;
        case(100) :  // d translate right
        {        translate_x += 0.03f;
        }  break;
        case(114) :  // r magnify move away
        {        translate_z -= 0.03f;
        }  break;
        case(102) :  // f shrink  move toward
        {        translate_z += 0.03f;
        }  break;
          
          
          
       }
    //printf("key = %d\n",key);  // p pause is 112
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float trans_speed = 0.0006f; 
    float rotate_speed = 0.002f; 
    // linear    speed = a abs(z) + c    c = 1,  a=speed(5)-1)/ 5
    trans_speed = trans_speed*(.8*abs(translate_z) +1.);

    bool printbutton =false ;
    float dx, dy;
    dx = (float)(x - mouse_old_x); //if (abs(dx)>.25) dx =0.0;
    dy = (float)(y - mouse_old_y); //if (abs(dy)>.25) dy = 0.0;

    if (mouse_buttons & 1)     // Rotate around x and y axis pitch and yaw
    {
        rotate_x += dx * rotate_speed;
        rotate_y += dy * rotate_speed;
        if (printbutton) printf("mouse button 1 rotate x,y %g %g \n",rotate_x,rotate_y);

    }
    else if (mouse_buttons & 2) // magnification  z axis move push down on scroll button and move mouse
    { if (printbutton) printf("mouse button 2 translate %g %g %g\n",translate_x,translate_y,translate_z);
        translate_z += dy * trans_speed;
    }
    else if(mouse_buttons & 4)    // Translate side to side or up and down
    { if (printbutton) printf("mouse button 4 %g %g %g\n",translate_x,translate_y,translate_z);
        translate_x += dx * trans_speed;
        translate_y -= dy * trans_speed;}

    else if(mouse_buttons & 3)
    { if (printbutton) printf("mouse button 3\n");}
    else if(mouse_buttons & 0)
    { if (printbutton) printf("mouse button 0\n");}
    //else 
    //   printf(" else mouse button = %d\n",mouse_buttons);

    mouse_old_x = x;
    mouse_old_y = y;
}



//Fancy cuda kernel can be called using dev_P, dev_MM, dev_DD   but define it with local names
// move<<<  >>> ( pos,dev_P,dev_MM,dev_DD);



////////////////////////////////////////////////////////////////////////
///////////////////  move3d  ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
//cuda kernel with four meshes and depth for 3d UVW's read from Field files
//uses MM[0:2] for the UVW and MM[2] to provide angle and depth
// move3d<<<  >>> ( pos,dev_P,dev_MM,dev_DD);

__global__
void move3d(float4 *pos, struct PPart *PP,struct MMesh *MM, struct DData *DD, curandState_t* states){
// Cuda Kernal to move the particles
// loop on all particles using cudaindex and stride
// for each particle find i_ele, depth angle     findiele
// interpolate sigma coordinate, find three corner values, average them to PP[iP].xyz
// Did that with all three time steps. Time interpolate
// Step PP[iP] position forward.

int IpTest=-250;
//int DeBuG = false;   //   true or false

/*   real stuff now  */
double dt_sec=DT_SEC;
//float deg2pi = 3.1415926/180.;

//  Cuda strides
int cudaindex = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

// Main time loop. Loop CUDA_STEPS times between returns for plotting
double  time_now = DD[0].time_now;

// initialize seed for the random macro defined in main.h: 
// #define RANDP 987.8353*randP - int(987.8353*randP) -.5
//float randP = abs(time_now/.2348579723 - int(time_now/.2348579723))  ;
float randP = (32.235643324*time_now + 3.454)-int(32.235643324*time_now + 3.454);
randP = .5555;


for (int itime=0; itime<CUDA_STEPS; itime++){
        
    for(int Ip = cudaindex; Ip <NUM_PARTICLES; Ip += stride){

// Update Particle information.  PP.age PP.state
        PP[Ip].age++;
        float Wf = PP[Ip].WfCore; //*cos((time_now- PP[Ip].WfShft)*PP[Ip].WfFreq );
        
        if (time_now > PP[Ip].Release_time  && PP[Ip].state==3 ){
            // Wait is over start moving
            PP[Ip].state = 1;  // move
            PP[Ip].x_present = PP[Ip].XYZstart[0];
            PP[Ip].y_present = PP[Ip].XYZstart[1];
            PP[Ip].z_present = PP[Ip].XYZstart[2];

        }
        if (PP[Ip].age>60*57600 && PP[Ip].state==1){
            //  Been moving for 2 days, put into wait mode again at end of line
            PP[Ip].state = 3;
            //PP[Ip].age += - MAX_GLPARTICLES +3400;   // assuming the first batch was separated by 1 dt each
            PP[Ip].age =  - 10800;   // two hours
            PP[Ip].x_present = PP[Ip].XYZstart[0];
            PP[Ip].y_present = PP[Ip].XYZstart[1];
            PP[Ip].z_present = PP[Ip].XYZstart[2];
            
        }
        /**/   //This thing was broken, needed to set the i_ele to starting value 
        if (PP[Ip].state == 2  && false) {
            // grounded particle start right up again
            PP[Ip].state = 1;  // move
            PP[Ip].age = 0;
            // Start right up, with age measured from old Release_time + duration  it was moving
            //    ie age is always time_now-time_init -Release_time
            PP[Ip].Release_time = time_now-MM[0].time_init;

            PP[Ip].x_present = PP[Ip].XYZstart[0];
            PP[Ip].y_present = PP[Ip].XYZstart[1];
            PP[Ip].z_present = PP[Ip].XYZstart[2];
            PP[Ip].i_ele = 55;
            for (int i=0; i<4; i++) PP[Ip].i_ele4[i] = 55;
            
        }
        /**/

// move particle
        if (PP[Ip].state == 1 ) { // move particle

        // Find surrounding triangle of Particle for all three meshes
        //  PP[Ip].i_ele4[iMM]
        //  PP[Ip].factor4[iMM][0:2]
        for (int iMM=0; iMM<3; iMM++) 
        {
          
                findiele(Ip,iMM,PP,MM); 

        }  
        PP[Ip].answer[0]=0.0;

        if (Ip==IpTest && itime==0) printf(" move3d finished findiele %d\n",itime);
        // interpolate values for angle and depth at PP[Ip].x,y    
        float VAR[3];
        int iMM=2;    //  mesh for w, angle and depth
        for (int i=0; i<3; i++) 
        { // i_ele is the element, ele[i_ele[0:2] are the nodes at corners of triangle i_ele
            long elei = MM[iMM].ele[PP[Ip].i_ele4[iMM]][i];
            VAR[i]=MM[iMM].ANGLE[elei];
        }
        if (Ip==IpTest && itime==0) printf("move3d before Interpolate2D findiele itime= %d, time_now= %fs %gh\n",itime,time_now,time_now/3600.);
        Interpolate2D(Ip,iMM,PP,VAR); 
        float angle=PP[Ip].answer[0];
        //or
        iMM=2; for (int i=0; i<3; i++) VAR[i]=MM[iMM].depth[MM[iMM].ele[PP[Ip].i_ele4[iMM]][i]];
        if (Ip==IpTest) printf(" depths = %g %g %g \n",VAR[0],VAR[1],VAR[2]);
        Interpolate2D(Ip,iMM,PP,VAR);  
        float depth=PP[Ip].answer[0]; 
        if (Ip==IpTest && itime==0) printf("move3d after Interpolate2D angle[%d]=%g  depth=%g\n",Ip,angle,depth);

        // Find zeta, sea surface. 
        // Pick out the three DD's to interpolate in time
        int DDT0=DD[0].DD3[0];
        int DDT1=DD[0].DD3[1];
        int DDT2=DD[0].DD3[2];
        
        iMM=2; for (int i=0; i<3; i++) VAR[i]=DD[DDT0].zeta[MM[iMM].ele[PP[Ip].i_ele4[iMM]][i]];
        Interpolate2D(Ip,iMM,PP,VAR); float Z0=PP[Ip].answer[0];
        iMM=2; for (int i=0; i<3; i++) VAR[i]=DD[DDT1].zeta[MM[iMM].ele[PP[Ip].i_ele4[iMM]][i]];
        Interpolate2D(Ip,iMM,PP,VAR); float Z1=PP[Ip].answer[0];
        iMM=2; for (int i=0; i<3; i++) VAR[i]=DD[DDT2].zeta[MM[iMM].ele[PP[Ip].i_ele4[iMM]][i]];
        Interpolate2D(Ip,iMM,PP,VAR); float Z2=PP[Ip].answer[0];
        
        float time_frac=(time_now - DD[DDT0].time)/(DD[DDT2].time - DD[DDT0].time);
        //float a =  2.*vart[2] -4.*vart[1] +2.*vart[0];
        //float b = -   vart[2] +4.*vart[1] -3.*vart[0];
        //float c =                             vart[0];
        //float Upnow = a*time_frac*time_frac + b*time_frac + c;
        float ZETA =  ( 2.*Z2 -4.*Z1 +2.*Z0)*time_frac*time_frac 
        +(   -Z2 +4.*Z1    -3*Z0)*time_frac 
        +(                  Z0);   
        PP[Ip].Sigma = (ZETA- PP[Ip].z_present)/(ZETA+depth) ;
        //if ((Ip==17150 || Ip==17151) && itime<2) {
        //    printf("ZETA SIGMA Test i=%d Z= %g, ZETA=%g d=%g Sigma=%g\n"
        //    ,Ip, PP[Ip].z_present, ZETA,depth,PP[Ip].Sigma); }

        
        // Find the isigmap, isigmam and sigmafrac 
        //  U[iP] = U[isigmap]*sigmafrac +U[isigmam]*(1.0-sigmafrac)
        // do three times and use timefrac to produce final VAR[3] at corners
        // iMM = 0U  1V  2W   
        // Only works for UVW.  In future add special cases 3T 4S iMM=iMM-1 or -2 

        iMM=0; Interpolatesigma(Ip, iMM, PP, DD, MM, depth,ZETA,time_now); 
        float Up=PP[Ip].answer[0];
        iMM=1; Interpolatesigma(Ip, iMM, PP, DD, MM, depth,ZETA,time_now); 
        float Vp=PP[Ip].answer[0];
        float cosa = cos(angle);
        float sina = sin(angle);
        float Upnow = cosa*Up -sina*Vp;
        float Vpnow = sina*Up +cosa*Vp;

        iMM=2; Interpolatesigma(Ip, iMM, PP, DD, MM, depth,ZETA, time_now); 
        float Wpnow=PP[Ip].answer[0];

        if (Ip==IpTest && itime==0) printf("move3d after sigma UVp[%d]= %g %g UVWpnow= %g, %g, %g  angle=%g  depth=%g\n"
           ,Ip,Up,Vp,Upnow, Vpnow, Wpnow,angle,depth);

        
        /*  Now have time and space interpolates of U,V,W for particle */
        /* Apply them to the particle coordinates and done! 
        (unless temporal runge kutta is needed. 
            Running goofy small time steps)*/
            float KH = MM[0].KH;   // Random jiggle 100 / sqrt(3600/1.5)   So 3600/1.5 * KH = 100
            float KV = MM[0].KV;   // Random jiggle 100 / sqrt(3600/1.5)   So 3600/1.5 * KH = 100
            //  KH and KV contain the sqrt(DT_SEC) for time stepping random walk
            
            randP = curand_normal(&states[cudaindex]);
            PP[Ip].x_present += dt_sec*(Upnow*1.) +(randP-.0)*KH; 
            randP = curand_normal(&states[cudaindex]);
            PP[Ip].y_present += dt_sec*(Vpnow*1.) +(randP-.0)*KH; 
            randP = curand_normal(&states[cudaindex]);
            PP[Ip].z_present += dt_sec*Wpnow*1.0  +Wf  +(randP-.0)*KV;
            
            PP[Ip].z_present = min(PP[Ip].z_present, ZETA );       // if z_p is above -0.01
            PP[Ip].z_present = max(PP[Ip].z_present, -depth);    // if z_p is below -depth
            //if (Ip==17150 && itime<2) {
            //    printf("i=%d ZETA=%g, z_pre=%g\n"
            //    ,Ip, ZETA, PP[Ip].z_present); }
         // end of if PP[Ip].state = 1  moving particle updated
        }
          
            // End of Particle loop on all Ip
        }
        
        // End of a time step, increment to next  time_now += dt_sec;
        // if time_frac >1, then it will fall out of the loop and not increment PP.timenow
        time_now+=dt_sec;    

    }
    // Update the VBO  pos[]
    for(int Ip = cudaindex; Ip <NUM_PARTICLES; Ip += stride){
        int Ipx = Ip%MAX_GLPARTICLES;       // Not too many and only from moveable points
        if (PP[Ip].state == 0)
            {// white boundary 
            // Set ColorClass to 1.0
            pos[Ipx] = make_float4(PP[Ip].x_present,PP[Ip].y_present,PP[Ip].z_present,  0.0f);
            }
        else if(PP[Ip].state == 2) 
            {// Aground   place at zero zero origin
            // Set ColorClass to 1.0
            pos[Ipx] = make_float4(0.0f,0.0f,0.05f,  1.0f);
            }
        else if(PP[Ip].state == 1) 
            {// regular moving point
             // Set ColorClass to float value between 0.0-6.0 
             // To accommodate states 0 and 2, add 2.0 to push to 2-8   
            // all modes work with shaderpipe.vs

                float ColorClass;
                double NumColors = 6.; 
            if (MM[0].color_mode == 0) { // ColorByRelease
                double agesec = PP[Ip].Release_time - MM[0].time_init;
                ColorClass = (agesec/MM[0].age_class) ; 
                if (ColorClass>NumColors) ColorClass=NumColors;
            }
            else if (MM[0].color_mode == 1) { // ColorByAge
                double agesec = time_now- PP[Ip].Release_time;
                ColorClass = (agesec/MM[0].age_class) ; //% NumColors;
                if (ColorClass>NumColors) ColorClass=NumColors;
            }
            else if (MM[0].color_mode == 2) { // ColorByPulse
                double agesec = PP[Ip].Release_time - MM[0].time_init;
                //ColorClass = floor(agesec/(MM[0].pulse_spacing)) ; //% NumColors;
                ColorClass = agesec/(10.*MM[0].pulse_spacing) ; //% NumColors;
                while (ColorClass > NumColors) ColorClass-=NumColors; //% NumColors;
            }
            else if (MM[0].color_mode == 3) {// ColorByDepth
                // with Zeta depth can be positive.  Make sure code is greater than 1.0 , 
                //  1.0 -> 0.0 interval should be first colorclass. 
                ColorClass = max(1.0-(PP[Ip].z_present/MM[0].depthcolorinterval),0.0); 
                if (ColorClass>NumColors) ColorClass=NumColors;
            }
            else if (MM[0].color_mode == 4) {// ColorByOrigin
                // Really just color by Latitude of XYZstart with offset and scaling in meters
                ColorClass = (PP[Ip].XYZstart[1]+315000.)/10000.; 
                while (ColorClass < 0.0) ColorClass+=10000.; //% NumColors;
                while (ColorClass > NumColors) ColorClass-=NumColors; //% NumColors;
            }
            else if (MM[0].color_mode == 5) {// ColorBySigma
                // Sigma values expanded out to 1 : 6.  to give 1. 1.5 2. 2.5 etc..
                ColorClass = (PP[Ip].Sigma)*NumColors; 
                if (ColorClass < 0.0) ColorClass=0.; //% NumColors;
                if (ColorClass > NumColors) ColorClass=NumColors; //% NumColors;
            }


            // Add 2 to ColorClass so that cases for 0 and 1 are accommodated
            pos[Ipx] = make_float4(PP[Ip].x_present,PP[Ip].y_present,PP[Ip].z_present,  ColorClass+2. );

        }     
    }

    // end of move()
    if ( cudaindex==0) DD[0].time_now = time_now;   // Only update dev_DD[] once
    //  Hopefully the other cudas have gotten started by now and don't need to read dev_DD[0].time_now
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
// update which triangle the iP is in for the three meshes   iMM 012 UVW  
//           note the iMM=2 also gives iele and factors for ANGLE and depth
// Sets PP[Ip].i_ele4[iMM] and the factors  PP[Ip].factor4[iMM][0:2] 
// for (in iMM=0; iMM<3; iMM++)
__device__ void findiele(int Ip,int iMM,struct PPart *PP, struct MMesh *MM)
{


    int i_ele, keepgoing, k;
    float xpart, ypart;
    float smallest_value = -0.05000; //  -.01  -0.001; 

    // Find surrounding triangle of Particle
    i_ele = PP[Ip].i_ele4[iMM];
    xpart = PP[Ip].x_present;
    ypart = PP[Ip].y_present;
    //return;
    //if(Ip==0) printf(" start findiele i_ele=%d \n",i_ele);
    //  Check for out of domain/ grounded particle
    //  do work if in-domain  else increment igrounded and skip main part of move
    if (i_ele >= 0 && PP[Ip].state==1) { 
        
        keepgoing = 1; 
        while (keepgoing  > 0 ){
            
            //  if any of the f's are negative, walk that way and restart while loop
            k=0;
            PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + 
            MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
            if ( PP[Ip].factor4[iMM][k] < smallest_value) { 
                i_ele = MM[iMM].tri_connect[i_ele][0]; 
            }
            else { 
                k=1;
                PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
                if ( PP[Ip].factor4[iMM][k] < smallest_value ) { 
                  i_ele = MM[iMM].tri_connect[i_ele][1] ; 
            }
            else { 
                k=2;
                PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
                if ( PP[Ip].factor4[iMM][k] < smallest_value ) { 
                i_ele = MM[iMM].tri_connect[i_ele][2] ;
                }
                else {
                   //  Found it, iele,   all f's are positive 
                   keepgoing = 0;
            }
            }
         }
         if (i_ele < 0) {    // newly grounded particle, zero him out.
               PP[Ip].state = 2;   // set state = grounded 
               PP[Ip].factor4[iMM][0]=0.0; PP[Ip].factor4[iMM][1]=0.0; PP[Ip].factor4[iMM][2]=0.0;
               PP[Ip].i_ele4[iMM] = i_ele;
               keepgoing = 0;
         }
         if (keepgoing>0) keepgoing++;
         if (keepgoing > 7000) { 
             printf(" k%d  ",Ip);
             PP[Ip].state = 2;   // set state = grounded 

         i_ele=-1;
         PP[Ip].i_ele4[iMM] = -1;
         PP[Ip].x_present=0.0;
         PP[Ip].y_present=0.0;
         PP[Ip].z_present=0.0;

         keepgoing=0;}
       }   
       
       //return;
       if (i_ele>=0){     // good particle still in the mesh
        PP[Ip].i_ele4[iMM]=i_ele;}
       // end of while keepgoing 

      // did it finish in a good element?    if not !good  ground it. 
      //  if (MM[iMM].goodele[i_ele])  PP[Ip].state = 2; 

}      
    return;
}



/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
// find 2d interpolated ANGLE(icase==0), depth(icase==1)
//  input is X,Y, A of i_ele points along with factor4
// MM[iMM].X[i_ele4[0:2]] MM[iMM].Y[i_ele4[0:2]] MM[iMM].ANGLE[i_ele4[0:2]] 
// PP[Ip].factor4[iMM][i_ele[0:]]    
// 2Dinterpolate(Ip,iMM,PP,MM,icase);  // icase = 0U, 1V, 2W, 3ANGLE, 4depth  
//    maybe do  VAR[3] = MM[iMM].ANGLE[i_ele4[0:2]]  instead of icase 
//    That way we can feed it the vertical interpolates of UVW[3]
//float VAR[3];
//iMM=3; for (int i=0; i<3; i++) VAR[i]=MM[iMM].angle[PP[Ip].iele4[iMM][i]];
//float angle = 2Dinterpolate(Ip,iMM,PP,MM,VAR);
//iMM=4; for (int i=0; i<3; i++) VAR[i]=MM[iMM].depth[PP[Ip].iele4[iMM][i]];
//float depth = 2Dinterpolate(Ip,iMM,PP,MM,VAR); 

__device__ void Interpolate2D(int Ip, int iMM, struct PPart *PP, float *VAR)
{


    float factor0=PP[Ip].factor4[iMM][0];
    float factor1=PP[Ip].factor4[iMM][1];
    float factor2=PP[Ip].factor4[iMM][2];
    
   PP[Ip].answer[0] = factor0*VAR[0]+factor1*VAR[1]+factor2*VAR[2];

}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

__device__ void Interpolatesigma(int Ip, int iMM, 
    struct PPart *PP, struct DData *DD, struct MMesh *MM, float depth, float ZETA, float time_now )
{
    // Find the isigmap, isigmam and sigmafrac 
    //  U[iP] = U[isigmap]*sigmafrac +U[isigmam]*(1.0-sigmafrac)
    // do three times and use timefrac to produce final VAR[3] at corners
    // iMM = 0U  1V  2W   
    // Only works for UVW.  In future add special cases 3T 4S iMM=iMM-1 or -2     int DDT0, DDT2;
    int IpTest = -250;

    int i_ele = PP[Ip].i_ele4[iMM];   
    float vart[3];
    float var[3]; 
    int sp, sm;

    float sigIp = max( min(PP[Ip].z_present / (depth-ZETA) , -0.01) , -0.99);    // 0 to -1.0
    //float sigIp = PP[Ip].z_present / depth ;    // 0 to -1.0
    //  count up in sp to walk down in depth
    sp=1;
    while(MM[iMM].sigma[sp]< sigIp) sp++;  // increment if sp is still above sigIp
    sm = sp-1;                         // sp is below sigIp,  sm is above
    float sigfrac = (sigIp-MM[iMM].sigma[sp])/(MM[iMM].sigma[sm]- MM[iMM].sigma[sp]);
    
    // Pick out the three DD's to interpolate in time
    int DD3[3];
    DD3[0]=DD[0].DD3[0];
    DD3[1]=DD[0].DD3[1];
    DD3[2]=DD[0].DD3[2];
    int DDT0=DD3[0];
    //DDT1=DD3[1];
    int DDT2=DD3[2];
    
    if (Ip==IpTest  ) printf(" start of interpretsigma iMM=%d  z_present= %g /depth=%g =sigIP = %g \n  sm,sp sigma[%d]=%g sigma[%d]=%g sigIP %g sigfrac %g\n"
    ,iMM,PP[Ip].z_present, depth,sigIp,sm,MM[iMM].sigma[sm],sp,MM[iMM].sigma[sp],sigIp,sigfrac);
    // loop on time DD3[i]
    // loop on three corners ele[i_ele][j]
    // average sm and sp at the corner
    
    for (int it=0; it<3; it++){  // time loop for DD3[it]
        for (int j=0; j<3; j++){     // loop around corners to get sigma averaged variable
            long ele0=MM[iMM].ele[i_ele][j];
            if      (iMM==0){ // U
                var[j] = DD[DD3[it]].U[sm][ele0]*sigfrac  + DD[DD3[it]].U[sp][ele0]* (1.0 - sigfrac);
            }
            else if (iMM==1){ // V    
                var[j] = DD[DD3[it]].V[sm][ele0]*sigfrac  + DD[DD3[it]].V[sp][ele0]* (1.0 - sigfrac);
            }
            else if (iMM==2){ // W    
                var[j] = DD[DD3[it]].W[sm][ele0]*sigfrac  + DD[DD3[it]].W[sp][ele0]* (1.0 - sigfrac);
            }
            else { printf(" \n\n Bad iMM in Interpolatesigma %d\n\n",iMM); }
        }
        // Have sigma average var[0:2] at the three corners
        //interpolate to center, to get three time increments vart[0:2] 
        vart[it]= PP[Ip].factor4[iMM][0]*var[0] 
        + PP[Ip].factor4[iMM][1]*var[1]
        + PP[Ip].factor4[iMM][2]*var[2];
        
        if (Ip==IpTest ) printf("  intersig DD3=%d var=%g %g %g vart=%g\n "
             ,DD3[it],var[0],var[1],var[2],vart[it]);
    }
    // Finally interpolate in time to get final answer for U, V, W to mover PP[Ip]
    // float time_now = DD[0].time_now;    // Will use dev_DD after the first pass with new DD
    float time_frac=(time_now - DD[DDT0].time)/(DD[DDT2].time - DD[DDT0].time);
    
    //float a =  2.*vart[2] -4.*vart[1] +2.*vart[0];
    //float b = -   vart[2] +4.*vart[1] -3.*vart[0];
    //float c =                       vart[0];
    //float Upnow = a*time_frac*time_frac + b*time_frac + c;
float Upnow = ( 2.*vart[2] -4.*vart[1] +2.*vart[0])*time_frac*time_frac 
             +(-   vart[2] +4.*vart[1] -3.*vart[0])*time_frac 
             +(                      vart[0]);                



/*  Now have time sigma and space interpolates of U,V,W for particle */
PP[Ip].answer[0] = Upnow;
        if (Ip==IpTest ) printf("  intersigend timenow=%fs timefrac=%g Upnow=%g\n ",time_now,time_frac,Upnow);


}