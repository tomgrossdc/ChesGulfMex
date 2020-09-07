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
    cout<<"Pipell "<<CC.Pipell[0]<<" "<<CC.Pipell[1]<<endl;
    
    MMesh *MM;
    MM =  (MMesh *)malloc(4*sizeof(MMesh));
    
    
    /*filetemplates  examples:
    "/media/tom/MyBookAllLinux/NOSnetcdf/201912/nos.cbofs.regulargrid.20191207.t18dz.n006.nc",
    "/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc",
    */
    
    //int year = 2020, month=02, day=7, hour= 18;  //, minute=5;  // 19/5/12/1/5

    tm today = {}; 
    today.tm_year =CC.year-1900 ; 
    today.tm_mon = CC.month-1;
    today.tm_mday = CC.day ;       
    today.tm_hour = CC.hour ;   // To make file name agree, else hours 0,1,2  map to 1,2,3 
    time_t ToDay = timegm(&today);   // conversion to UTC, not like mktime which is local
    char fps[256];
    strftime(fps,80, "  mainpart startup  timegm(&today) = %A %G %b %d   %R ", gmtime(&ToDay));
    cout<<fps<<endl;
    //ToDay = mktime(&today);
    //strftime(fps,80, "  mainpart startup  mktime(&today) = %A %G %b %d   %R ", gmtime(&ToDay));
    //cout<<fps<<endl;

    MM[0].ToDay = ToDay;
    //MM[0].ToDay = today;


//    MM[0].filetemplate = CC.filetemplate;
//    string newername = NetCDFfiledate(MM[0].filetemplate,MM);
//    cout<< " newername from NETCDFfiledate(MM)"<< newername <<endl;
    MM[0].filetemplate = CC.filetemplate;
    string newername;
    if(CC.IsFVCOM)
        { newername = NetCDFfiledateG(MM[0].filetemplate,MM);}
    else 
        { newername = NetCDFfiledate(MM[0].filetemplate,MM);}
    cout<< " newername from NETCDFfiledate(MM)"<< newername <<endl;

    
    

        //   icase-1 is used to include Regular call during testing.  Fix later...
        for (int icase=1; icase<4; icase++){

         int iMM=icase-1;   //  fill up MM[0] with U, MM[1] with V etc.
         printf(" \n\n Read Lat and Lon and Set MM[%d]\n",icase-1);
         if(CC.IsFVCOM) 
            {ReadMeshFieldG(newername,icase,MM);}
         else
            {ReadMeshField(newername,icase,MM);}

            printf(" Convert Lat Lon to Meters   node= %d \n",node);
            float DEG_PER_METER= 90./(10000*1000);
            for (int i=0;i<node;i++) {
               MM[iMM].X[i] = (( MM[iMM].Lon[i]-CC.LONmid) /DEG_PER_METER )*cos(CC.LATmid * PI/180.);
               MM[iMM].Y[i] =  ( MM[iMM].Lat[i]-CC.LATmid) /DEG_PER_METER;
               }

        cout<<endl<<"Launch delaunator and build MM["<<iMM<<"].ele, tri_connect, a_frac "<<endl;

         MakeMeshEle(iMM, MM);
         node = MM[icase-1].node;
         nsigma = MM[icase-1].nsigma;

         // Identify interior elements as MM[iMM].goodele[iele]=true;
         for (int i=0; i< MM[iMM].nele; i++){
             MM[iMM].goodele[i]=true;
            if (MM[iMM].ele[i][0]>MM[iMM].firstnodeborder) MM[iMM].goodele[i]=false;
            if (MM[iMM].ele[i][1]>MM[iMM].firstnodeborder) MM[iMM].goodele[i]=false;
            if (MM[iMM].ele[i][2]>MM[iMM].firstnodeborder) MM[iMM].goodele[i]=false;
         }
         }


    
    /*
    printf("\nmain  after set_Mesh_MMESH  node %d  MM[0].node %d\n"
        , node, MM[0].node);
        int iMM=0; 
        //int i_ele; 
        for (iMM=0; iMM<4; iMM++){for (int iP=0; iP<MM[iMM].node; iP+=5000)
            printf(" main Mesh set MM[%d].depth[%d] = %g  MM.ANGLE = %g\n",
            iMM,iP, MM[iMM].depth[iP], MM[iMM].ANGLE[iP]);
        }
   */
    
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

// Set some CC data into DD
//    DD[0].filetemplate = MM[0].filetemplate;
    for (int i=0; i<4; i++) 
        {
            DD[i].filetemplate = CC.filetemplate;
            DD[i].IsFVCOM = CC.IsFVCOM;
            DD[i].IsNGOFS = false;   // gets set to true when ww read fails
        }   

    //checkGpuMem();
    float free_m,total_m,used_m;
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);
    free_m =(uint)free_t/1048576.0 ;
    total_m=(uint)total_t/1048576.0;
    used_m=total_m-free_m;
    printf ( "  CUDA mem Free %ld b = %f Mb \n  CUDA mem total %ld b = %f Mb mem used %f Mb\n\n"
    ,free_t,free_m,total_t,total_m,used_m);
    
    /* 
    *  end of cuda memory / structure  setups
    */
    
    // Initialize some DD data, 
    // Zero out all Velocities to initialize
    //  Makes the coastline capture grounded particles, as it is not read as a velocity
    DD[0].ToDay =MM[0].ToDay;
    //DD[0].ToDay = timegm(&today);
    DD[0].ToDay -= 7200;    // two hours to correct for += below, and some other hour
    for(int ifour = 0; ifour<4; ifour++){
        for (int i=0; i<NODE; i++){
            for(int isig=0; isig<NSIGMA; isig++){
                DD[ifour].U[isig][i]=0.0;
                DD[ifour].V[isig][i]=0.0;
                DD[ifour].W[isig][i]=0.0;
            }
        }
        DD[0].ToDay +=3600;  // for hourly files
 
        //string newername = NetCDFfiledate(DD[0].filetemplate,DD);
        //ReadDataRegNetCDF(newername,ifour,DD,MM);
        //ReadFieldNetCDF(newername,ifour, DD, MM);
        string newername;
        //ReadDataRegNetCDF(newername,ifour,DD,MM);
        if (CC.IsFVCOM)
        {   newername = NetCDFfiledateG(DD[0].filetemplate,DD);
            cout<<"IsFVCOM  newername= "<<newername<< endl;
            ReadFieldNetCDFG(newername,ifour, DD, MM);}
        else
        {   newername = NetCDFfiledate(DD[0].filetemplate,DD);
            cout<<"IsNOTFVCOM  newername= "<<newername<< endl;
            ReadFieldNetCDF(newername,ifour, DD, MM);}


        printf("ReadData finished,  DD[%d].time=%f sec time=%g hr \n\n",ifour,DD[ifour].time,DD[ifour].time/3600.);
        //int id = 50; int isig=2;
        //printf("ReadData DD[%d].time %fs %ghr \n  DD[%d].UVW[%d][%d] %g %g %g \n  MM[0].XY[%d]= %g  %g\n\n",
        //  ifour, DD[ifour].time,DD[ifour].time/3600,
        //  ifour,isig, id,DD[ifour].U[isig][id],DD[ifour].V[isig][id],DD[ifour].W[isig][id],
        //  id,MM[0].X[id],MM[0].Y[id]);
    }    
    cout<<endl;
    time_now = (DD[0].time + DD[1].time)/2.;   // timefrac = .25
    for (int i=0; i<4; i++) DD[i].time_now = time_now; 
    MM[0].time_init = time_now;
    MM[0].Time_Init = MM[0].ToDay;
    MM[0].Time_Init += 1800; 
    printf(" mainpart  MM[0].time_init = %f DD[0].time = %f \n",MM[0].time_init,DD[0].time);
//    char fps[256];
    strftime(fps,80, " mainpart  MM[0].Time_Init = %A %G %b %d   %R ", gmtime(&MM[0].Time_Init));
    cout<<fps<<endl;


    for (int i=0; i<4; i++) DD[0].DD3[i]=i;



    /*      
    Build the Particle Struct PPart host_P
    */
    
    printf("\nInitialize the PPart Structs \n");
    
    size_t PPSizeGeneral ;
    PPSizeGeneral = sizeof(PPart)*num_P;        
    host_P = (PPart *)malloc(PPSizeGeneral);
    
    
    //  Elaborate this routine for different inital conditions of particles
    num_P = MM[0].node-MM[0].firstnodeborder +CC.NUM_PARTICLEs;
    if (num_P> NUM_PARTICLES) num_P=NUM_PARTICLES;
    
    PPartInit(host_P,MM,&CC,num_P);
    cout << " CC.shadervs = "<< CC.shadervs << endl;
    cout << " CC.shaderfs = "<< CC.shaderfs << endl;
    cout << " CC.filetemplate ="<<CC.filetemplate<<endl;
    MM[0].shadervs=CC.shadervs;
    MM[0].shaderfs=CC.shaderfs;
    MM[0].run_mode=CC.run_mode;
    MM[0].color_mode=CC.color_mode;
    MM[0].Dot_Size = CC.Dot_Size; 
    MM[0].depthcolorinterval = CC.depthcolorinterval;
    MM[0].age_class = CC.age_class;
    MM[0].pulse_spacing = CC.pulse_spacing;
    MM[0].time_init = time_now;
    MM[0].KH=CC.KH;
    MM[0].KV=CC.KV;
    

    for (int ip=0; ip<num_P; ip++) host_P[ip].time_now=time_now;
    
    
    //MMesh *dev_MM;    // no need to do this. Space is cudaMalloc'd and call is to (struct MMesh dev_MM)
    size_t MMSizeGeneral = 4*sizeof(MMesh);
    cudaMalloc((void**)&dev_MM,MMSizeGeneral);
    cudaMemcpy(dev_MM,MM,MMSizeGeneral,cudaMemcpyHostToDevice);
    
    //cudaMemcpy(dev_MM,MM,MMSizeGeneral,cudaMemcpyHostToDevice);
    
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


