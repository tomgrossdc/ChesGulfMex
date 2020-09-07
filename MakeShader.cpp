/* MakeShader.cpp 
Subroutines to access shader.fs and shader.vs


main output of this is gWVPLocation  which is used by RenderSceneCB 
with:       glUniformMatrix4fv(gWVPLocation, 1, GL_TRUE, (const GLfloat*)p.GetWVPTrans());

notice hardwired file names shader.vs and shader.fs  


*/
#ifndef WIN32
#include <unistd.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <string.h>
#include <assert.h>
// #include "ogldev_types.h"
#include <fstream>

#include <iostream>
using namespace std;

#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

//#include "ogldev_util.h"
//#include "ogldev_camera.h"
//#include "ogldev_pipeline.h"

#include "CControl.h"

//const char* pVSFileName = "shader.vs";
//const char* pFSFileName = "shader.fs";
const char* pVSFileName = "shader.vs";
const char* pFSFileName = "shader.fs";

GLuint gWVPLocation;

bool ReadFileNew(string pFileName, string& outFile);

static void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType)
{
    GLuint ShaderObj = glCreateShader(ShaderType);

    if (ShaderObj == 0) {
        fprintf(stderr, "Error creating shader type %d\n", ShaderType);
        exit(1);
    }

    const GLchar* p[1];
    p[0] = pShaderText;
    GLint Lengths[1];
    Lengths[0]= strlen(pShaderText);
    glShaderSource(ShaderObj, 1, p, Lengths);
    glCompileShader(ShaderObj);
    GLint success;
    glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar InfoLog[1024];
        glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
        fprintf(stderr, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
        exit(1);
    }

    glAttachShader(ShaderProgram, ShaderObj);
}

void CompileShaders(string shadervs, string shaderfs)
{
    printf(" CompileShaders \n");
    cout<<"shadervs = "<<shadervs<<endl;
    cout<<"shaderfs = "<<shaderfs<<endl;

GLenum err = glewInit();
if (err != GLEW_OK)
  exit(1); // or handle the error in a nicer way
if (!GLEW_VERSION_2_1)  // check that the machine supports the 2.1 API.
  exit(1); // or handle the error in a nicer way

    GLuint ShaderProgram = glCreateProgram();

    if (ShaderProgram == 0) {
        fprintf(stderr, "Error creating shader program\n");
        exit(1);
    }


    string vs, fs;
//    if (!ReadFileNew(pVSFileName, vs)) {
    if (!ReadFileNew(shadervs, vs)) {
        exit(1);
    };
    //cout<< pVSFileName <<endl;
    //cout<< vs<<endl;

//    if (!ReadFileNew(pFSFileName, fs)) {
    if (!ReadFileNew(shaderfs, fs)) {
        exit(1);
    };
    //cout << fs<<endl;

    AddShader(ShaderProgram, vs.c_str(), GL_VERTEX_SHADER);
    AddShader(ShaderProgram, fs.c_str(), GL_FRAGMENT_SHADER);

    GLint Success = 0;
    GLchar ErrorLog[1024] = { 0 };

    glLinkProgram(ShaderProgram);
    glGetProgramiv(ShaderProgram, GL_LINK_STATUS, &Success);
	if (Success == 0) {
		glGetProgramInfoLog(ShaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
	}

    glValidateProgram(ShaderProgram);
    glGetProgramiv(ShaderProgram, GL_VALIDATE_STATUS, &Success);
    if (!Success) {
        glGetProgramInfoLog(ShaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }

    glUseProgram(ShaderProgram);

    gWVPLocation = glGetUniformLocation(ShaderProgram, "gWVP");
    assert(gWVPLocation != 0xFFFFFFFF);
}

bool ReadFileNew(string pFileName, string& outFile)
{
    ifstream f(pFileName);
    
    bool ret = false;
    
    if (f.is_open()) {
        string line;
        while (getline(f, line)) {
            outFile.append(line);
            outFile.append("\n");
        }
        
        f.close();
        
        ret = true;
    }
    else {
        cout<<" Error in MakeShader.cpp ReadFileNew reading: "<<pFileName<<endl;
    }
    
    return ret;
}


/*
void transmatrix(float transmatrix[], float tx,float ty, float tz)
{
    float trans[16]={1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,tx,ty,tz,1.};  //translate

    for (int i=0; i<16; i++){
        transmatrix[i] = trans[i];
    //return transmatrix;
    }
}

float* Matrix4();
GLfloat* idmatrix();
void transmatrix(float *matrix, float tx,float ty, float tz);
GLfloat* zrotmatrix(float angle);
GLfloat* xrotmatrix(float angle);


float* Matrix4() 
{
    float matrix[16] = {1.0,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.};  //Identity
    return matrix;
}

 GLfloat* idmatrix(){
       GLfloat matrix[16] = {1.0,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.};  //Identity
    return matrix;
    }


GLfloat* zrotmatrix(float angle){
    float ca=cos(angle); float sa=sin(angle);
    GLfloat zrotmatrix[16] = {ca,sa,0.,0.,-sa,ca,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.};  //rotate around z axis
    return zrotmatrix;
    }

GLfloat* xrotmatrix(float angle){
    float ca=cos(angle); float sa=sin(angle);
    GLfloat xrotmatrix[16] = {1.,0.,0.,0.,0.,ca,sa,0.,0.,-sa,ca,0.,0.,0.,0.,1.};  //rotate around x axis
    return xrotmatrix;
    }
    */


/*
float * matrix_trans(float oldmatrix[16], float tx,float ty, float tz)
{
    //static float trans[16]={1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,tx,ty,tz,1.};  //translate
    static float trans[16]={1.,0.,0.,tx,0.,1.,0.,ty,0.,0.,1.,tz,0.,0.,0.,1.};  //translate
    static float transout[16];
    
    for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        transout[ij4(i,j)] = trans[ij4(i,0)] * oldmatrix[ij4(0,j)] +
                         trans[ij4(i,1)] * oldmatrix[ij4(1,j)] +
                         trans[ij4(i,2)] * oldmatrix[ij4(2,j)] +
                         trans[ij4(i,3)] * oldmatrix[ij4(3,j)];
       //printf("tranout[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),trans[ij4(i,j)]);
       }
    }
    printf(" matrix_transout %g %g %g\n",transout[3],transout[13],transout[7]);
    return transout;
}
*/


//////////////////////////////////////////////////////////////////////
inline int ij4(int i, int j)
    {
       return (j*4+i);
    };
////////////////////         matrix_RotTrPer    //////////////////////
//////////////////////////////////////////////////////////////////////
float * matrix_RotTrPer(float old0[16], 
            float anglex, float angley,float anglez,
            float tx, float ty, float tz,
            float width, float height, float znear, float zfar, float FOV)
{  // Combined Rotation, Translation, Perspective
    // Does not include camera translation rotation
    static float transout[16];
    static float old1[16];
    //static float old2[16];
    float ca,sa;
    int CASE_ORDER[7]{0,1,2,3,4};   // rotx=0,roty=1, rotz=2, rottrans=3, rotperspective=4, rotplustrans=5, rotnegtrans=6 
    //int CASE_ORDER[7]{6,0,5,1,2,3,4};   // rotx=0,roty=1, rotz=2, rottrans=3, rotperspective=4, rotplustrans=5, rotnegtrans=6 
    //static float trans[16];  
for (int icase=0; icase<5; icase++) {
if (CASE_ORDER[icase]==0 )   // rot around x axis
    {
    ca=cos(anglex);
    sa=sin(anglex);
    float trans[16]={1.,0.,0.,0.,0.,ca,sa,0.,0.,-sa,ca,0.,0.,0.,0.,1.};  
        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old1[ij4(i,j)] = trans[ij4(i,0)] * old0[ij4(0,j)] +
                         trans[ij4(i,1)] * old0[ij4(1,j)] +
                         trans[ij4(i,2)] * old0[ij4(2,j)] +
                         trans[ij4(i,3)] * old0[ij4(3,j)];
       //printf("rotxout[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),trans[ij4(i,j)]);
        } }
        for (int i=0; i<16; i++) old0[i]= old1[i]; 
    }
else if(CASE_ORDER[icase]==1) // rot around y axis
    {
    ca=cos(angley);
    sa=sin(angley);
    float trans[16]={ca,0.,sa,0.,0.,1.,0.,0.,-sa,0.,ca,0.,0.,0.,0.,1.};  
        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old1[ij4(i,j)] = trans[ij4(i,0)] * old0[ij4(0,j)] +
                         trans[ij4(i,1)] * old0[ij4(1,j)] +
                         trans[ij4(i,2)] * old0[ij4(2,j)] +
                         trans[ij4(i,3)] * old0[ij4(3,j)];
       //printf("rotyout[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),trans[ij4(i,j)]);
        } }     
        for (int i=0; i<16; i++) old0[i]= old1[i]; 
    }
else if(CASE_ORDER[icase]==2)  //  rot around z axis
    {
    ca=cos(anglez);
    sa=sin(anglez);
    float trans[16]={ca,sa,0.,0.,-sa,ca,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.};  
    
        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old1[ij4(i,j)] = trans[ij4(i,0)] * old0[ij4(0,j)] +
                         trans[ij4(i,1)] * old0[ij4(1,j)] +
                         trans[ij4(i,2)] * old0[ij4(2,j)] +
                         trans[ij4(i,3)] * old0[ij4(3,j)];
       //printf("rotzout[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),trans[ij4(i,j)]);
        } } 
        for (int i=0; i<16; i++) old0[i]= old1[i]; 
    }
else if (CASE_ORDER[icase]==3)  //  translation
{

    float trans[16] = {1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,tx,ty,tz,1.};  //translate

        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old1[ij4(i,j)] = trans[ij4(i,0)] * old0[ij4(0,j)] +
                         trans[ij4(i,1)] * old0[ij4(1,j)] +
                         trans[ij4(i,2)] * old0[ij4(2,j)] +
                         trans[ij4(i,3)] * old0[ij4(3,j)];
       //printf("trans[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),old2[ij4(i,j)]);
        } } 
        for (int i=0; i<16; i++) old0[i]= old1[i]; 
    }

else if (CASE_ORDER[icase]==4)  // perspective
{
//  width,height,znear,zfar,FOV
// float width, float height, float znear, float zfar, float FOV)
    const float aspect = width / height;
    const float zrange = znear - zfar;
    const float tanHalfFOV = tanf((FOV / 2.0)*3.141/180.);

    float trans[16] = {1.0f / (tanHalfFOV * aspect),0.,0.,0.
    ,0.,1.0f / tanHalfFOV,0.,0.
    ,0.,0.,(-znear-zfar)/zrange,   2.0f*zfar*znear/zrange
    ,0.,0.,1.,0.};  
        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old1[ij4(i,j)] = trans[ij4(i,0)] * old0[ij4(0,j)] +
                         trans[ij4(i,1)] * old0[ij4(1,j)] +
                         trans[ij4(i,2)] * old0[ij4(2,j)] +
                         trans[ij4(i,3)] * old0[ij4(3,j)];
       //printf("perspective[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),old1[ij4(i,j)]);
        } } 
        for (int i=0; i<16; i++) old0[i]= old1[i]; 

    }

else if (CASE_ORDER[icase]==5)  //  positive partial translation
{

    float trans[16] = {1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,ty,0.,1.};  //translate with tz=0

        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old1[ij4(i,j)] = trans[ij4(i,0)] * old0[ij4(0,j)] +
                         trans[ij4(i,1)] * old0[ij4(1,j)] +
                         trans[ij4(i,2)] * old0[ij4(2,j)] +
                         trans[ij4(i,3)] * old0[ij4(3,j)];
       //printf("trans[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),old2[ij4(i,j)]);
        } } 
        for (int i=0; i<16; i++) old0[i]= old1[i]; 
    }
        
else if (CASE_ORDER[icase]==6)  //  negative partial translation
{

    float trans[16] = {1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,-ty,0.,1.};  //translate

        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old1[ij4(i,j)] = trans[ij4(i,0)] * old0[ij4(0,j)] +
                         trans[ij4(i,1)] * old0[ij4(1,j)] +
                         trans[ij4(i,2)] * old0[ij4(2,j)] +
                         trans[ij4(i,3)] * old0[ij4(3,j)];
       //printf("trans[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),old2[ij4(i,j)]);
        } } 
        for (int i=0; i<16; i++) old0[i]= old1[i]; 
    }

}    
    return old0;
}



/*
//////////////////////////////////////////////////////////////////////
////////////////////         matrix_rotxyz    ////////////////////////
//////////////////////////////////////////////////////////////////////
float * matrix_rotxyz(float oldmatrix[16], 
                      float anglex, float angley,float anglez)
{// rotate, transform, perspective
    static float transout[16];
    static float old[16];
    float ca,sa;

    //static float trans[16];  
for (int icase=0; icase<3; icase++) {
if (icase==0 )   // rot around x axis
    {
    ca=cos(anglex);
    sa=sin(anglex);
    float trans[16]={1.,0.,0.,0.,0.,ca,sa,0.,0.,-sa,ca,0.,0.,0.,0.,1.};  
        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        transout[ij4(i,j)] = trans[ij4(i,0)] * oldmatrix[ij4(0,j)] +
                         trans[ij4(i,1)] * oldmatrix[ij4(1,j)] +
                         trans[ij4(i,2)] * oldmatrix[ij4(2,j)] +
                         trans[ij4(i,3)] * oldmatrix[ij4(3,j)];
       //printf("rotxout[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),trans[ij4(i,j)]);
        } } 
    }
else if(icase==1) // rot around y axis
    {
    ca=cos(angley);
    sa=sin(angley);
    float trans[16]={ca,0.,sa,0.,0.,1.,0.,0.,-sa,0.,ca,0.,0.,0.,0.,1.};  
        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        old[ij4(i,j)] = trans[ij4(i,0)] * transout[ij4(0,j)] +
                         trans[ij4(i,1)] * transout[ij4(1,j)] +
                         trans[ij4(i,2)] * transout[ij4(2,j)] +
                         trans[ij4(i,3)] * transout[ij4(3,j)];
       //printf("rotyout[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),trans[ij4(i,j)]);
        } }     
    }
else if(icase==2)  // icase ==2   // rot around z axis
    {
    ca=cos(anglez);
    sa=sin(anglez);
    float trans[16]={ca,sa,0.,0.,-sa,ca,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.};  
    
        for (unsigned int i = 0 ; i < 4 ; i++) {
       for (unsigned int j = 0 ; j < 4 ; j++) {
        transout[ij4(i,j)] = trans[ij4(i,0)] * old[ij4(0,j)] +
                         trans[ij4(i,1)] * old[ij4(1,j)] +
                         trans[ij4(i,2)] * old[ij4(2,j)] +
                         trans[ij4(i,3)] * old[ij4(3,j)];
       //printf("rotzout[%d][%d]= [%d] = %g\n",i,j,ij4(i,j),trans[ij4(i,j)]);
        } } 
    }
}    
    return transout;
}
*/
