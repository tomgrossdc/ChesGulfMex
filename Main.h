/***************************************************************************
 *   Copyright (C) 2004 by                                       	   *
 *  Tom Gross                                                              *
 *                                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

//Test how to check out.

#ifndef MAIN_H
#define MAIN_H

//  WARNING Change these do make clean
#define NODE 175000     // 16384 g 8150 8050 8000  x 8100 8200
#define NROWS 128      //  g 32
#define NSIGMA 41
#define NELE 1050000    //  g 15800   175000 1050000 2000000
#define NELE3 2000000
#define NUM_PARTICLES  1000000   //150000   250000 65000dec4  1500625 g1625 1600 x 1650 1700 1800 2000
#define MAX_GLPARTICLES 1000000 //150000   250000 65000
//#define DT_SEC 1.5             // Wierd error requires this to be set in CControl.h
#define CUDA_STEPS 20   // 60=3sec/hr    10=6sec/hr 
#define FILE_DTIME 3600.
#define SCALE_GL 1.0;  //  do in shader 0.0000075  // .000015(full bay) scale PP.x_ to pos[Ipx][4]  
//#define LON_LAT_KEY_00 -76.00
//#define LON_LAT_KEY_01  38.00


//#define RANDP (1000.*(PP[Ip].x_present+PP[Ip].y_present) - int(1000.*(PP[Ip].x_present+PP[Ip].y_present))) -.5
//#define RANDP (1000.*UPNOW - int(1000.*(UPNOW))) -.5
#define RANDP (35.68493*randP + 56.309293084) - int(35.68493*randP + 56.309293084) 


#include <iostream>
using namespace std;
#include <fstream>
#include <string.h>
#include <math.h>
#include <iomanip>
#include <time.h>
#include <thread> 
#include <stdlib.h>

//#include <netcdf>
#include "netcdf"
using namespace netCDF;
using namespace netCDF::exceptions;




//#include <GL/glew.h>
#include <GL/freeglut.h>

//#include "netcdfcpp.h"
//#include "netcdf.h"#include "date.h" 

#include "MMesh.h"
#include "DData.h"
#include "PParticle.h"
//#include "MakeShader.h"

//#include "move.h"

//#include "dump_netcdf.h"

//#include "date.h"

//#include "Bugs.h"

#define PI 3.14159265358979

// The original code of triangle.h and triangle.c is written by C.
//extern "C" {
//	#include "triangle.h"
//}

/*  //these are global from mainpart.cu
    // all are reset in code to real values
    //  The define NODE, NUM_PARTICLES are
    // mainly used only to guarantee excess space in struc's

    double dt_sec;
    int num_P = NUM_PARTICLES;
    int node = NODE; 
    int nsigma = NSIGMA;  
    float time_now;
*/

struct PPart { 
    float answer[4];
    int p_id;
    int i_ele;
    int i_ele4[4]; 
    float factor4[4][3];
    float x_present; 
    float y_present; 
    float z_present; 
    float time_now;
    float factor[3];
    int num_P;
    int state;   // white boundary=0; moving=1; grounded=2; waiting=3
    long age;     // units of DT_SEC   
    float Release_time;
    float XYZstart[3];
    float WfCore=0.0;      // Wf=WfCore*cos(time_now*WfFreq - WfShft);
    float WfFreq=0.0;      //  2*pi/(Period sec)
    float WfShft=0.0;
    float Sigma=.5;        // ZETA/depth

    };

struct MMesh {
    float Lon[NODE];
    float Lat[NODE];	
    float X[NODE];
    float Y[NODE];
    float ANGLE[NODE];
    float Xbox[4];
	float depth[NODE];
	float sigma[NSIGMA];
	float factor[3];	
	float a_frac[NELE][3];	
	float b_frac[NELE][3];
	float c_frac[NELE][3];	
 	long tri_connect[NELE][3];	
	long ele[NELE][3];
    bool goodele[NELE];
    int node;
    int nsigma;
    int nele;
    float Mask[352737];
    time_t ToDay;
    int firstnodeborder;
    char* filetemplate;
    string shadervs;
    string shaderfs;
    string run_mode;
    int color_mode;
    float age_class; 
    float pulse_spacing;
    float pulse_duration;
    float depthcolorinterval=5.;
    int runmode = 0;
    int Dot_Size;
    float time_init;
    time_t Time_Init;
    float KH;
    float KV; 
};

struct DData {
    double time;
    float U[NSIGMA][NODE];
    float V[NSIGMA][NODE];
    float W[NSIGMA][NODE];
    float temp[NSIGMA][NODE];
    float salt[NSIGMA][NODE];
    float Kh[NSIGMA][NODE];
    float zeta[NODE];
    int DD3[4];
    time_t ToDay;
    float time_now;
    char* filetemplate;
    bool IsFVCOM = false;
    bool IsNGOFS = false;


};
void Cuda_Move();
void ReadMesh(string& filename, struct MMesh *MM);  //regular read, not tested since Jan 2020

void ReadMeshField(string& filename, int icase, struct MMesh *MM);
string NetCDFfiledate(char* filenametemplate, struct MMesh *MM);
string NetCDFfiledate(char* filenametemplate,struct DData *MM);

void ReadMeshFieldG(string& filename, int icase, struct MMesh *MM);
string NetCDFfiledateG(char* filenametemplate, struct MMesh *MM);
string NetCDFfiledateG(char* filenametemplate,struct DData *MM);

// MakeShader.cpp
bool ReadFileNew(const char* pFileName, string& outFile);
static void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType);
void CompileShaders();
//void transmatrix(float transmatrix[], float tx,float ty, float tz);
float * matrix_trans(float oldmatrix[16], float tx,float ty, float tz);
float * matrix_rotxyz(float oldmatrix[16], float ax,float ay, float az);
float * matrix_RotTrPer(float old0[16], 
            float anglex, float angley,float anglez,
            float tx, float ty, float tz,
            float width, float height, float zNear, float zFar, float FOV);


//extern struct CControl CC;


#include "CControl.h"

#endif
