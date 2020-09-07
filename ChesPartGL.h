// ChesPartGL.h  
//  all subroutine prototypes
// and global variables
// and GL includes

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>

#include <GL/freeglut.h>

// Not needed until add new mouse routines
// Throws numerous  __host__ annotation warnings during compilation
#include <glm/glm.hpp>
#include <glm/ext.hpp>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
//#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 1024; //620;  // 1024;  //512
const unsigned int window_height = 1024;  // 1024;  //512
GLdouble znear = 0.50;
GLdouble zfar = 10.0;

//const unsigned int mesh_width    = 256;
//const unsigned int mesh_height   = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
static GLuint gWVPLocation;     // static makes it visible to MakeShader.cpp



float g_time_now = 0.0;
int iDD = -1;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 1.57;  // 1.57, 1.57  not so good
float translate_z = -7.0;
float translate_y = 0.0;
float translate_x = 0.0;
int Dot_Size;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

// global data structures
PPart *host_P;
PPart *dev_P;
MMesh *dev_MM;
MMesh MM[4];
DData *dev_DD;
DData DD[4];
int DD3[4];
//std::thread t1;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool GLmoveparticle(struct PPart *PP, struct MMesh *MM, 
                    struct DData *DD); //, struct CControl *CC);
void cleanup();

// GL functionality
//bool initGL(int *argc, char **argv);
bool initGL();
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

void ReadData(double time_now, int ifour, DData *DD, MMesh *MM);

const char *sSDKsample = "ChesPartGL (VBO)";
// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);

//int mainChesPartGL(struct PPart *PP, struct MMesh *MM, struct DData *DD);
struct TimeNow { float g_time_now ; };
__global__ void move(float4 *pos, struct PPart *PP,struct MMesh *MM, struct DData *DD);
__global__ void move3d(float4 *pos, struct PPart *PP,struct MMesh *MM, struct DData *DD,curandState_t* states);


__device__ void findiele(int Ip,int iMM,struct PPart *PP, struct MMesh *MM);   
__device__ void Interpolate2D(int Ip, int iMM, struct PPart *PP, float *VAR);
__device__ void Interpolatesigma(int Ip, int iMM, 
   struct PPart *PP, struct DData *DD, struct MMesh *MM, float depth, float ZETA, float time_now );

__device__ void findielefake(int Ip,int iMM,struct PPart *PP, struct MMesh *MM);
