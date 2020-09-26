/*-------------------------------------------------------------------
//
// C++ Interface: main
//
// Description: mainpart.cpp, main.h
//   The C++ program, mainpart.cpp will read the starting lon, lat, start_time,
//   end_time and the inputfilename. The geometry will be read off the netcdf file
//   and mesh parameters calculated and stored into RAM.  The main line program will
//   loop over the time variable.  The velocity fields will be read off the NetCDF
//   file only when necessary to update the time stepping. An inner loop will go over
//   all particles.  Parallelization of the run can be done at this point by simply
//   calculating groups of particles on multiple processors.  
//
//
// Author: Tom Gross 
//
// Copyright: See COPYING file that comes with this distribution
//
//
---------------------------------------------------------------------*/

#include "Main.h"

/**/
    double dt_sec;
    int num_P = NUM_PARTICLES;
    int node = NODE; 
    int nsigma = NSIGMA;  
    float time_now;
/**/

#include "ChesPartGL.h"
#include "FindElei.h"

#ifndef SIMP_GLCU
#define SIMP_GLCU
#include "ChesPartGL.cu"
#endif

//#include "struct.h"
// try to add a global struct for reference to internal routines
// Struct these at end of ChesPartGL.h 

//PPart *host_P;
//PPart *dev_P;
//MMesh *dev_MM;
//DData *dev_DD;

extern struct CControl CC;


int  main( int argc, char** argv ) {
    printf("mainpart Arguments: %d\n",argc);
    for (size_t i{}; i<argc; i++) {
        printf(" argv[%zd] = %s\n",i,argv[i]);
    }
    
    
    printf("mainpart.cu  Cuda based particle mover \n");
    
    /*----------------------------------------------------------------------
    // Read in all the time independent data and put them into MM struct.
    ----------------------------------------------------------------------*/  
    CControl CC;
    char st0[128]="dummydummydummy"; CC.filetemplate=st0;
    bool error;
    if(argc==1){
        error = CC.read_control_data("CControl/CBOFS_ColorbyDepth.txt");
        }
        else {
        error = CC.read_control_data(argv[1]);
        }
    cout << "error =" << error << endl;
    cout<<"CC.Pipell "<<CC.Pipell[0]<<" "<<CC.Pipell[1]<<endl;
    
    
    
    
    MMesh *MM;
    MM =  (MMesh *)malloc(4*sizeof(MMesh));
    

    BuildMM(MM, CC) ;

         
    
 //  Little routine to find the node number of center of 
// range as set in CC.  
float lonpp=CC.LONmid+CC.LONwidth*0.;
float latpp=CC.LATmid+CC.LAThieght;
int iMM=0;
int inode = FindElei(lonpp,latpp,MM, iMM );
printf("\n FindElei NEWEST Lon=%g Lat=%g inode=%d MM[%d].Lon=%g MM[%d].Lat=%g \n",
lonpp,latpp, inode, iMM, MM[iMM].Lon[inode],iMM, MM[iMM].Lat[inode]);   




    // No need to initialize DD here.  ReadData will do that later followed by cudaMemcpy
    printf("\n\nFour separate DData's for past present future and reading\n");
    DData *DD;
    size_t DDSizeGeneral = sizeof(DData)*4;
    DD =  (DData *)malloc(DDSizeGeneral);



// Start of BuildDD(struct DData *DD, struct MMesh *MM, struct CControl CC);


BuildDD(DD,MM,CC);



/*      
    Build the Particle Struct PPart host_P
*/
    
    printf("\nInitialize the PPart Structs \n");
    
    
    size_t PPSizeGeneral ;
    PPSizeGeneral = sizeof(PPart)*num_P;        
    host_P = (PPart *)malloc(PPSizeGeneral);
    
    num_P = MM[0].node-MM[0].firstnodeborder +CC.NUM_PARTICLEs;
    if (num_P> NUM_PARTICLES) num_P=NUM_PARTICLES;
    
PPartInit(host_P,MM,&CC,num_P);
     
    
    
    //MMesh *dev_MM;    // no need to do this. Space is cudaMalloc'd and call is to (struct MMesh dev_MM)
    size_t MMSizeGeneral = 4*sizeof(MMesh);
    cudaMalloc((void**)&dev_MM,MMSizeGeneral);
    cudaMemcpy(dev_MM,MM,MMSizeGeneral,cudaMemcpyHostToDevice);
    
    
    //PPart *dev_P;
    //size_t PPSizeGeneral = (sizeof(PPart)*num_P);
    cudaMalloc((void**)&dev_P,PPSizeGeneral);
    cudaMemcpy(dev_P,host_P,PPSizeGeneral,cudaMemcpyHostToDevice);
    //printf("after cudaMalloc for dev_P, PPSizeGeneral=%ld\n\n",PPSizeGeneral);
    
    //DData *dev_DD;
    cudaMalloc((void**)&dev_DD,DDSizeGeneral);
    cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);
    
    
    printf("/n/nLaunch GLmoveparticle from mainpart.cu\n");
    
    #if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
    #endif
    
    cout <<" CC.shadervs = "<< CC.shadervs << endl;
    
    GLmoveparticle(host_P,MM,DD); //, &CC);
    
    
    
    /*----------------------------------------------------
    // End of particle movement calculation.
    ----------------------------------------------------*/   
	printf("\n mainpart.cp END \n");
    cudaFree(dev_DD);
    cudaFree(dev_MM);
    cudaFree(dev_P);
    
    
    return 0;
} // end of main    


