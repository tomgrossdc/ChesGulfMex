#include "Main.h"
#include "DData.h"

#include "math.h"

/*----------------------------------------------------------
  struct.h   
  collection of structure based things to include 
  in mainpart.cu

/*------------------------------------------------------------*/    
void ReadData(double time_now, int ifour, DData *DD, MMesh *MM)
{

int node = NODE;   // MM[0].node ?
int nsigma = NSIGMA;   // MM[0].nsigma  ?
//printf("ReadData   node = %d, MM[0].node= %d \n",node,MM[0].node);
//printf("ReadData   nsigma = %d, MM[0].nsigma= %d \n",nsigma,MM[0].nsigma);

// Read new tranch of data into DD[ifour]
double Radius, Phi;
float Ur,Vr,Wr,ctime,stime,wctime;
float U_base = 2.;
float V_base = 2.;
float W_base = .1;
double pi = 3.1415926;
double phitime = 2.908882037e-4; //2pi/3600/6 six hourly

phitime = 2.*pi/3600./8.;

DD[ifour].time = time_now;
ctime = cos((time_now*phitime));   // tidal cycle
stime = sin((time_now*phitime));   // tidal cycle
wctime=cos((time_now*phitime*6.));  // faster cycle for w wave
ctime=ctime/.15;
printf(" ReadData time_now =%g   ctime = %g MM[0].node=%d\n"
   ,time_now,ctime,MM[0].node);

  for (int i=0; i< node ; i++){
for (int jbad = 0; jbad<1; jbad++){
	Radius = pow(sqrt(MM[0].X[i]*MM[0].X[i] + MM[0].Y[i]*MM[0].Y[i]),.75);
	Phi = atan2(MM[0].Y[i],MM[0].X[i]) ;
	//Ur = U_base * cos(Phi)*ctime +0.0;
	//Vr = V_base * sin(Phi)*stime +0.0;
  Ur = -U_base * (2.*pi*Radius/(3600.*12.)) * cos(pi/2.-Phi)*ctime;
	Vr = V_base * (2.*pi*Radius/(3600.*12.)) * sin(pi/2.-Phi)*ctime;
  Wr = W_base * ctime*cos(MM[0].X[i]*pi/MM[0].Xbox[1])*1.0;
  //Ur = -U_base * cos(pi/2.-Phi)*ctime +0.0;
	//Vr =  V_base * sin(pi/2.-Phi)*ctime +0.0;
	//Wr =  W_base * cos(Radius/10000.)*ctime;

  for (int iz = 0; iz< nsigma; iz++){
    DD[ifour].U[iz][i]=-Ur;
    DD[ifour].V[iz][i]=-Vr;
    DD[ifour].W[iz][i]= Wr;
    DD[ifour].temp[iz][i]=20.;
    DD[ifour].salt[iz][i]=35.;
    DD[ifour].Kh[iz][i]=0.01;
  }
}//jbad
  } 
}



void ReadDataRegNetCDF(string& filename, int ifour, DData *DD, MMesh *MM)
{
// declarations of arrays for reading data are below LLL, LL
long ij, ij2, isigm;
int nx,ny;
int node, nsigma, Depth, numtime; 
//printf("  ReadData  "+filename);
cout<<" ReadDataRegNetCDF: " << filename << endl;

NcDim dim;

NcFile dataFile(filename, NcFile::read);

   NcVar data=dataFile.getVar("u_eastward");
   if(data.isNull()) printf(" data.isNull u_eastward/n");
   for (int i=0; i<4; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) numtime=dimi;
      if (i==1) Depth = dimi;
      if (i==2) ny = dimi;
      if (i==3) nx = dimi;

      //cout<<"ifour="<<ifour<<" i= "<<i<<" dimi "<<dimi<<endl;
   }
//printf("dimi  ifour=%d  numtime=%d,Depth=%d,ny=%d,nx=%d total=%d\n",ifour,numtime,Depth, ny,nx, numtime*Depth*ny*nx*5);
float LLL[numtime][1][ny][nx];

// Cant dimension this to 15*693*509 = 26455275  so do one sigma level at time
// float* LLLL= new float
//float LLLL[5291055];

// Declare start Vector specifying the index in the variable where
// the first of the data values will be read.
std::vector<size_t> start(4);

start[0] = 0;
start[1] = 0;
start[2] = 0;
start[3] = 0;

// Declare count Vector specifying the edge lengths along each dimension of
// the block of data values to be read.
std::vector<size_t> count(4);

count[0] = 1;
count[1] = 1;
count[2] = ny;
count[3] = nx;


// loop over sigma coordinate up to Depth times
for (start[1]=0; start[1]<Depth; start[1]++) {
   data.getVar(start,count,LLL);   
        isigm = start[1];
         ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
         {ij2=i*nx+j; if (MM[0].Mask[ij2]>.5){
         DD[ifour].U[start[1]][ij] = LLL[0][0][i][j]; ij++; } }  }
   }
   

   data=dataFile.getVar("v_northward");
   if(data.isNull()) printf(" data.isNull v_northward/n");
// loop over sigma coordinate by small groups of size Depth (=5)
for (start[1]=0; start[1]<Depth; start[1]++) {
   data.getVar(start,count,LLL);   
         ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
         {ij2=i*nx+j; if (MM[0].Mask[ij2]>.5){
         DD[ifour].V[start[1]][ij] = LLL[0][0][i][j]; ij++; } }  }
   }

//   WTF  There is no W velocity in the NetCDF file!
   //data=dataFile.getVar("w");
   //if(data.isNull()) printf(" data.isNull w/n");
// loop over sigma coordinate by small groups of size Depth (=5)
for (start[1]=0; start[1]<Depth; start[1]++) {
   //data.getVar(start,count,LLL);   
         ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
         {ij2=i*nx+j; if (MM[0].Mask[ij2]>.5){
         DD[ifour].W[start[1]][ij] = 0.0         ;   ij++; } }  }
    }



   // ocean_time:units = "seconds since 2016-01-01 00:00:00" ;
	//	ocean_time:calendar = "gregorian" ;
   NcVar datat=dataFile.getVar("ocean_time");
   if(datat.isNull()) printf(" datat.isNull ocean_time/n");
   double LL[10];  // Bigger array works better than *LL, same as LL[1]  double is float
   datat.getVar(LL);
   //   try to use older data files to fill in missing ones
   if(LL[0] > DD[ifour].time) 
      {
         DD[ifour].time = LL[0];   //Seconds 
      }
      else
      {
         DD[ifour].time +=3599. * 4. ;   // four hours since the previous value in this position
      }
   
dataFile.close();

//printf(" ReadDataRegNetCDF END   DD[%d].time = %g sec  %g hr\n\n"
//      ,ifour,DD[ifour].time,DD[ifour].time/3600.);
}





//////////////////////////////////////////////////////////////////////////////////////////////
//    Read the field netcdf files.  They have three types of meshes and new names.
//    The three meshes will be in X: MM[0], Y: MM[1], Z: MM[2]
//    Add a variable to the MM  
//	float u(ocean_time, s_rho, eta_u, xi_u) ;
//	float v(ocean_time, s_rho, eta_v, xi_v) ;
//	float w(ocean_time, s_w, eta_rho, xi_rho) ;
void ReadFieldNetCDF(string& filename,int ifour, DData *DD, MMesh *MM)
{   
    // MeshCode has the three variable names. 
    // u, v, w  is MeshCode[0]    icase=0, 1, 2   also MM[icase]
    //   angle MM[3] is already set by first mesh read

long ij, ij2;
int nx,ny;
int node, nsigma, Depth, numtime; 
NcDim dim;
float LLL[NODE];

std::vector<std::string> MeshCode={
            "u","v","w","angle",
        };

// open the netcdf file
cout<< "ReadFieldNetCDF "<<filename<<endl;
try {
   NcFile dataFile(filename, NcFile::read);

for (int icase=0; icase<3; icase++){
   //printf(" ReadFieldData ",icase);
// u, v, w  is MeshCode[0]    icase=0, 1, 2   also MM[icase]
   NcVar data=dataFile.getVar(MeshCode[icase]);
   if(data.isNull()) printf("ReadVariableNetCDF data.isNull /n");
   //printf("ReadFieldData DD[%d]  icase=%d dimi= ",ifour, icase);
   for (int i=0; i<4; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) numtime=dimi;
      if (i==1) nsigma = dimi;
      if (i==2) ny = dimi;
      if (i==3) nx = dimi;
      //printf(" %ld",dimi);
   }
   //printf("\n");
   nsigma = min(nsigma,NSIGMA);
   Depth = nsigma/5;
//float *LLL{new float[numtime*Depth*ny*nx]{}};  // 
//printf(" LLL = %ld %ld  \n",sizeof(LLL), sizeof(LLL)/sizeof(float));
int iLLL = 0; 
//float* LLL = new float{[numtime]{[Depth]{[ny]{[nx]}}}};  // 
//float LLL[numtime][Depth][ny][nx]= new float[numtime*Depth*ny*nx];

// Declare start Vector specifying the index in the variable where
// the first of the data values will be read.
std::vector<size_t> start(4);

start[0] = 0;
start[1] = 0;
start[2] = 0;
start[3] = 0;

// Declare count Vector specifying the edge lengths along each dimension of
// the block of data values to be read.
std::vector<size_t> count(4);

count[0] = 1;
count[1] = 1;   //Depth;
count[2] = ny;
count[3] = nx;
// loop over sigma coordinate by small groups of size Depth (nsigma/5)
for (start[1]=0; start[1]<nsigma; start[1]++) {
   try {
      //printf(" data.getVar %d  %ld \n",start[1],nsigma);
      data.getVar(start,count,LLL);   
      } catch (const std::exception& e){ printf(" error on data.getVar %s\n",e.what()); }
   for (int it=0; it<numtime; it++)
      {
         int isig = start[1];
         ij=0; 
         for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++) {  
            ij2=i*nx+j; 
            if (MM[icase].Mask[ij2]>.5){
               iLLL = 0*it*(Depth*ny*nx)+ 0*isig*(ny*nx) + i*(nx) + j;
               if(icase==0) DD[ifour].U[isig][ij] = LLL[iLLL]; 
               if(icase==1) DD[ifour].V[isig][ij] = LLL[iLLL];
               if(icase==2) DD[ifour].W[isig][ij] = LLL[iLLL]; 
               ij++;
            }
         }  }  // for i for j
    } // for it
}  // for start[1] the depth loop for sigma
}  // icase loop

// Read 2D array for Sea Surface height
/*
	float zeta(ocean_time, eta_rho, xi_rho) ;
		zeta:long_name = "free-surface" ;
		zeta:units = "meter" ;
		zeta:time = "ocean_time" ;
		zeta:grid = "grid" ;
		zeta:location = "face" ;
		zeta:coordinates = "lon_rho lat_rho ocean_time" ;
		zeta:field = "free-surface, scalar, series" ;
		zeta:_FillValue = 1.e+37f ;
*/
      int iMM=2;    // same as W arrays for MM.mask
      NcVar data=dataFile.getVar("zeta");
      if(data.isNull()) printf(" data.isNull h/n");
      for (int i=0; i<3; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) numtime=dimi;
      if (i==1) ny = dimi;
      if (i==2) nx = dimi;
   }
      printf("ZETA dims: %d %ld",numtime,ny,nx);

      data.getVar(LLL);
      ij=0; 
      for (int i=0; i<(ny); i++)
        {
         for (int j=0; j<(nx); j++)
            {  ij2=i*nx+j;
               if (MM[iMM].Mask[ij2]>.5) 
                  { DD[ifour].zeta[ij] = LLL[ij2]; 
                    ij++;
                  } 
            }  
        }
      




// ocean_time:units = "seconds since 2016-01-01 00:00:00" ;
//	ocean_time:calendar = "gregorian" ;
   NcVar datat=dataFile.getVar("ocean_time");
   if(datat.isNull()) printf(" datat.isNull ocean_time/n");
   double LL[10];  // Bigger array works better than *LL, same as LL[1]  double is float
   datat.getVar(LL);
   //   try to use older data files to fill in missing ones
   if(LL[0] > DD[ifour].time) 
      {
         DD[ifour].time = LL[0];   //Seconds 
      }
      else
      {
         DD[ifour].time +=3599. * 4. ;   // four hours since the previous value in this position
      }
printf(" ReadFieldNetCDF ocean_time LL[0]=%g DD[0].time=%g \n", LL[0],DD[0].time);   

// close the netcdf file
dataFile.close();
} catch (const std::exception& e) { 
   printf( "Error on openning ReadFieldNetCDF file\n %s\n\n",e.what()); 
   exit(1);   // not a very clean way to exit. Cuda Leaves resources hanging.
   }
}




/*
void ReadDataFieldNetCDF(string& filename, int ifour, DData *DD, MMesh *MM)
{
// declarations of arrays for reading data are below LLL, LL
long ij, ij2;
int nx,ny;
int node, nsigma, Depth, numtime; 
//printf("  ReadData  "+filename);
cout<<"ReadDataFieldNetCDF: " << filename << endl;

NcDim dim;

NcFile dataFile(filename, NcFile::read);

   NcVar data=dataFile.getVar("u_eastward");
   if(data.isNull()) printf(" data.isNull u_eastward/n");
   for (int i=0; i<4; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) numtime=dimi;
      if (i==1) Depth = dimi/5;
      if (i==2) ny = dimi;
      if (i==3) nx = dimi;

      //cout<<"ifour="<<ifour<<" i= "<<i<<" dimi "<<dimi<<endl;
   }
//printf("dimi  ifour=%d  numtime=%d,Depth=%d,ny=%d,nx=%d\n",ifour,numtime,Depth, ny,nx);
float LLL[numtime][Depth][ny][nx];

// Declare start Vector specifying the index in the variable where
// the first of the data values will be read.
std::vector<size_t> start(4);

start[0] = 0;
start[1] = 0;
start[2] = 0;
start[3] = 0;

// Declare count Vector specifying the edge lengths along each dimension of
// the block of data values to be read.
std::vector<size_t> count(4);

count[0] = 1;
count[1] = Depth;
count[2] = ny;
count[3] = nx;
// loop over sigma coordinate by small groups of size Depth (=5)
for (start[1]=0; start[1]<15; start[1]+=count[1]) {
   data.getVar(start,count,LLL);   
      for (int it=0; it<numtime; it++){
       for (int isig=0; isig<count[1]; isig++) {
         ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
         {ij2=i*nx+j; if (MM[0].Mask[ij2]>.5){
         DD[ifour].U[isig+start[1]][ij] = LLL[it][isig][i][j]; ij++; } }  }
       }
    }
}
    
   data=dataFile.getVar("v_northward");
   if(data.isNull()) printf(" data.isNull v_northward/n");
// loop over sigma coordinate by small groups of size Depth (=5)
for (start[1]=0; start[1]<15; start[1]+=count[1]) {
   data.getVar(start,count,LLL);   
      for (int it=0; it<numtime; it++){
       for (int isig=0; isig<count[1]; isig++) {
         ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
         {ij2=i*nx+j; if (MM[0].Mask[ij2]>.5){
         DD[ifour].V[isig+start[1]][ij] = LLL[it][isig][i][j]; ij++;
         } }  }
       }
    }
}

//   WTF  There is no W velocity in the NetCDF file!
   //data=dataFile.getVar("w");
   //if(data.isNull()) printf(" data.isNull w/n");
// loop over sigma coordinate by small groups of size Depth (=5)
for (start[1]=0; start[1]<15; start[1]+=count[1]) {
   //data.getVar(start,count,LLL);   
      for (int it=0; it<numtime; it++){
       for (int isig=0; isig<count[1]; isig++) {
         ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
         {ij2=i*nx+j; if (MM[0].Mask[ij2]>.5){
         DD[ifour].W[isig+start[1]][ij] = 0.0;  //LLL[it][isig][i][j]; 
         ij++;
         } }  }
       }
    }
}


   // ocean_time:units = "seconds since 2016-01-01 00:00:00" ;
	//	ocean_time:calendar = "gregorian" ;
   NcVar datat=dataFile.getVar("ocean_time");
   if(datat.isNull()) printf(" datat.isNull ocean_time/n");
   double LL[10];  // Bigger array works better than *LL, same as LL[1]  double is float
   datat.getVar(LL);
   //   try to use older data files to fill in missing ones
   if(LL[0] > DD[ifour].time) 
      {
         DD[ifour].time = LL[0];   //Seconds 
      }
      else
      {
         DD[ifour].time +=3599. * 4. ;   // four hours since the previous value in this position
      }
printf(" ReadDataFieldNetCDF ocean_time LL[0]=%g DD[0].time=%g \n", LL[0],DD[0].time);   

dataFile.close();

//printf(" ReadDataFieldNetCDF END   DD[%d].time = %g sec  %g hr\n\n"
//      ,ifour,DD[ifour].time,DD[ifour].time/3600.);

}
*/
