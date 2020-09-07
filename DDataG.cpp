#include "Main.h"
#include "DData.h"

#include "math.h"





//NGOFS FVCOM
//////////////////////////////////////////////////////////////////////////////////////////////
//    Read the field netcdf files.  They have three types of meshes and new names.
//    The three meshes will be in X: MM[0], Y: MM[1], Z: MM[2]
//    Add a variable to the MM  
/*ncdump -h nos.nwgofs.fields.20200701.n008.nc
	float u(time, siglay, nele) ;
		u:long_name = "Eastward Water Velocity" ;
		u:standard_name = "eastward_sea_water_velocity" ;
		u:units = "meters s-1" ;
		u:grid = "fvcom_grid" ;
		u:type = "data" ;
		u:coordinates = "time siglay latc lonc" ;
		u:mesh = "fvcom_mesh" ;
		u:location = "face" ;
*/

void ReadFieldNetCDFG(string& filename,int ifour, DData *DD, MMesh *MM)
{   
    // ifour is sort of the time, specifies DD[ifour] to load with UVWzeta
    // FVCOM does not use mask so MMesh is not needed in ReadFieldNetCDFG
    //https://tidesandcurrents.noaa.gov/ofs/ofs_animation.shtml?ofsregion=ng&subdomain=0&model_type=currents_nowcast


long ij, ij2;
int nx,ny;
int node, nsigma, Depth, numtime; 
NcDim dim;
float LLL[NODE];
NcVar data;

// open the netcdf file
cout<<endl<< "ReadFieldNetCDFG#1 DD["<<ifour<<"] "<<filename<<endl;
try {
   NcFile dataFile(filename, NcFile::read);

data=dataFile.getVar("u");
   if(data.isNull()) printf("ReadVariableNetCDF data.isNull /n");
   //printf("ReadFieldDataG#2 DD[%d]  u \n",ifour);
   for (int i=0; i<3; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) numtime=dimi;
      if (i==1) nsigma = dimi;
      if (i==2) node = dimi;
      //printf("dimi= %ld  ",dimi);
   }
   printf(" numtime=%d , nsigma=%d , node=%d \n",numtime,nsigma,node);
   nsigma = min(nsigma,NSIGMA);

// Declare start Vector specifying the index in the variable where
// the first of the data values will be read.
// Declare count Vector specifying the edge lengths along each dimension of
// the block of data values to be read.
std::vector<size_t> start(4);
std::vector<size_t> count(4);
start[0] = 0;
start[1] = 0;
start[2] = 0;
start[3] = 0;
count[0] = 1;
count[1] = 1;   //nsigma Depth;
count[2] = node;
for (int isid=0; isid<nsigma; isid++) {
   start[1]=isid;
   try {
      //printf(" u  data.getVar %d  %ld %ld\n",start[1],nsigma,node);
      data.getVar(start,count,LLL);   
      //data.getVar(start,count,DD[ifour].U[isid]);   
      } catch (const std::exception& e){ printf(" error on data.getVar %s\n",e.what()); }
   int isidd = nsigma-isid -1 ;
   for (int i=0; i<node; i++){
      DD[ifour].U[isidd][i]= LLL[i]  ;
   }
}

data=dataFile.getVar("v");
for (int isid=0; isid<nsigma; isid++) {
   start[1]=isid;
try {
      //printf(" v  data.getVar %d  %ld %ld\n",start[1],nsigma, node);
      data.getVar(start,count,LLL);   
      //data.getVar(start,count,DD[ifour].V[isid]);   
      } catch (const std::exception& e){ printf(" error on data.getVar %s\n",e.what()); }
   int isidd = nsigma-isid -1 ;
   for (int i=0; i<node; i++){
      DD[ifour].V[isidd][i]= LLL[i];
   }
}


/* test the before and after reads
for (int inode=5000; inode<20000; inode+=5000){
   printf("\n PRE-DD[%d].W[:][%d]=",ifour,inode);
      for (int isig=0; isig<nsigma; isig+=1){
      printf("%g ",DD[ifour].W[isig][inode] );
   }
   printf("\n");
}
*/

data=dataFile.getVar("ww");
   if(data.isNull()) 
   {
      printf("\n \n\nReadFieldNedCDFG getVar( ww )  data.isNull \n");
      printf(" Make =0.0;  synthetic W to be made in mainpart after triangulation.  \n\n\n");
      for (int i=0; i<node; i++){
         for(int isidd=0; isidd<nsigma; isidd++ ){
            DD[ifour].W[isidd][i]= 0.0;
         }
      }
   // Flag that W needs to be calculated in interpolatesigma()
      DD[ifour].W[0][node]= -1000. ; 
      DD[ifour].IsNGOFS = true;
   }
   else
   {
      // Regular W reading for FVCOM that include ww
      printf(" ww data.getVar %d %ld %ld\n",ifour,nsigma,node);
      for (int isid=0; isid<nsigma; isid++) {
         start[1]=isid;
         try {
               data.getVar(start,count,LLL);   
               //data.getVar(start,count,DD[ifour].W[isid]);   
         } 
         catch (const std::exception& e){ printf(" error on data.getVar %s\n",e.what());    }

         int isidd = nsigma-isid -1 ;
         for (int i=0; i<node; i++){
            DD[ifour].W[isidd][i]= LLL[i]*1.0;
         }
      }
   }



// zeta is new dimensions:
data=dataFile.getVar("zeta");
   if(data.isNull()) printf("ReadVariableNetCDF data.isNull /n");
   //printf("ReadFieldDataG#2 DD[%d]  zeta \n",ifour);
   for (int i=0; i<2; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) numtime=dimi;
      if (i==1) node = dimi;
      //printf("dimi= %ld  ",dimi);
   }
   printf("zeta numtime=%d ,  node=%d \n",numtime,node);
start[0] = 0;
start[1] = 0;
count[0] = 1;
count[1] = node;
try {
      //printf(" zeta data.getVar %d  %ld\n",start[1],node);
      //data.getVar(start,count,LLL);   
      data.getVar(DD[ifour].zeta);   
      } catch (const std::exception& e){ printf(" error on data.getVar %s\n",e.what()); }

int ip = 5000;
printf("i4= %d U[%d]= %g V=%g W=%g Z=%g\n",ifour,ip,DD[ifour].U[0][ip],
     DD[ifour].V[0][ip],DD[ifour].W[0][ip],DD[ifour].zeta[ip] );


//float time(time) ;
//		time:long_name = "time" ;
//		time:units = "days since 2013-01-01 00:00:00" ;
//		time:format = "defined reference date" ;
//		time:time_zone = "UTC" ;

   NcVar datat=dataFile.getVar("time");
   if(datat.isNull()) printf(" datat.isNull time/n");
   double LL[10];  // Bigger array works better than *LL, same as LL[1]  double is float
   datat.getVar(LL);
   //   try to use older data files to fill in missing ones
   if(LL[0] > DD[ifour].time) 
      {
         DD[ifour].time = LL[0]*24.*60.*60.;   //Seconds 
      }
      else
      {
         DD[ifour].time +=3599. * 4. ;   // four hours since the previous value in this position
      }
printf(" ReadFieldNetCDFG ocean_time LL[0]=%g days DD[0].time=%f sec\n", LL[0],DD[ifour].time);   

// close the netcdf file
dataFile.close();
} catch (const std::exception& e) { 
   printf( "Error on openning ReadFieldNetCDFG file\n %s\n\n",e.what()); 
   exit(1);   // not a very clean way to exit. Cuda Leaves resources hanging.
   }
}