/*   Alter this by eliminating netcdf from set_Mesh 
    include reading a simple text file for lon, lat, depth, sigma
*/


#include "MMesh.h"

/*--------------------------------------------------------------
    Class : Mesh
    Author : Tom Gross, Hong Lin
    Contact info : tom.gross@noaa.gov  hong.lin@noaa.gov
    Superclass : none
    Subclass : none    
    Required files : mesh.h
    Description : Read in the mesh data
       elefunc
       Create the triangle connectivity array  triconnect
       and create the a,b,c for the linear interpolations
       Transform from meters to degrees and back
--------------------------------------------------------------*/    

/*--------------------------------------------------------------------
   All temporary variables will start with t_, such as t_ele.
   All temporary index will start with i_, such as i_time_future. 
----------------------------------------------------------------------*/



///////////////////////////////////////////////////////
//////////////      REGULAR     ///////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////


void ReadMesh(string& filename, struct MMesh *MM)
{


NcDim dim;
long ij;
int nx,ny;
int node,nodemore;

cout<<"ReadMesh: " << filename << endl;
NcFile dataFile(filename, NcFile::read);

   NcVar data=dataFile.getVar("mask");
   if(data.isNull()) printf(" data.isNull Mask/n");
for (int i=0; i<2; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) ny = dimi;
      if (i==1) nx = dimi;

      cout<<"ReadMesh  i= "<<i<<" dimi= "<<dimi<<endl;
   }
double Mask[ny][nx];
double LLL[ny][nx];
//double LLat[ny][nx];
//std::vector<std::vector<double> > LLL( ny , std::vector<double> (nx));  
//std::vector<std::vector<double> > Mask( ny , std::vector<double> (nx));  


printf("sizeof LLL  %ld\n",sizeof(LLL));

printf("sizeof Mask %ld\n",sizeof(Mask));

   data.getVar(Mask);
   ij=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
    { MM[0].Mask[ij] = Mask[i][j]; ij++; }  }

   data=dataFile.getVar("Longitude");
   if(data.isNull()) printf(" data.isNull Longitude/n");
   data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[0].Lon[ij] = LLL[i][j]; ij++;} }  }


 node = ij;
 int summask=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++){
       if((i-1)>=0) {
                        summask += Mask[i-1][j];
         if((j+1)<node) summask += Mask[i-1][j+1];
         if((j-1)>=0  ) summask += Mask[i-1][j-1];
       }
       if((i+1)<node) {
                        summask += Mask[i+1][j];
         if((j+1)<node) summask += Mask[i+1][j+1];
         if((j-1)>=0  ) summask += Mask[i+1][j-1];
       }
       if((j+1)<node)   summask += Mask[i][j+1];
       if((j-1)>=0  )   summask += Mask[i][j-1];
      
       if (Mask[i][j]==0 && summask > 0 ){
 //         MM[0].Lat[node] = LLat[i][j]; 
          MM[0].Lon[node] = LLL[i][j];
          MM[0].depth[node]=0.0;
          node++;}  
          summask=0; 
      } }  // i,j loop

  data=dataFile.getVar("Latitude");
  if(data.isNull()) printf(" data.isNull Latitude/n");
  data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[0].Lat[ij] = LLL[i][j]; ij++;} }  }

 node = ij;
 MM[0].firstnodeborder=node;  // initialize the first border node

 summask=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++){
       if((i-1)>=0) {
                        summask += Mask[i-1][j];
         if((j+1)<node) summask += Mask[i-1][j+1];
         if((j-1)>=0  ) summask += Mask[i-1][j-1];
       }
       if((i+1)<node) {
                        summask += Mask[i+1][j];
         if((j+1)<node) summask += Mask[i+1][j+1];
         if((j-1)>=0  ) summask += Mask[i+1][j-1];
       }
       if((j+1)<node)   summask += Mask[i][j+1];
       if((j-1)>=0  )   summask += Mask[i][j-1];
      
       if (Mask[i][j]==0 && summask > 0 ){
 //         MM[0].Lat[node] = LLat[i][j]; 
          MM[0].Lat[node] = LLL[i][j];
          MM[0].depth[node]=0.0;
          node++;}  
          summask=0; 
      } }  // i,j loop

 //node = ij;
 //MM[0].firstnodeborder=node;  // initialize the first border node


   data=dataFile.getVar("h");
   if(data.isNull()) printf(" data.isNull h/n");
   data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[0].depth[ij] = LLL[i][j]; ij++;} }  }


    MM[0].node = node;
    printf(" masked Lon, Lat num ij = %ld\n",ij);


   /* Although labeled sigma, in Regulargrid this is Depth
   Apparently the regular grid gets rid of sigma variations
   and just uses fixed depths for the 3d
   */
   data=dataFile.getVar("Depth");
   if(data.isNull()) printf(" data.isNull Depth/n");
   dim = data.getDim(0);
   size_t dimi = dim.getSize();
   MM[0].nsigma=dimi;
   cout<<" ReadMesh Depth dimi="<<dimi <<endl;
   double sigma[dimi];
   data.getVar(sigma);
   for (int i=0; i<dimi; i++) MM[0].sigma[i]=sigma[i];



// not needed as LL Mask will be deleted on exit from this routine
//delete [] LLL;
//delete [] Mask;

dataFile.close();
printf(" ReadMesh                  node = %d\n", MM[0].node);
bool Readtxt = false;   // Regular file
AddOutsideLonLat(0,Readtxt, MM);
printf(" ReadMesh after AddOutside node = %d\n", MM[0].node);

}

void AddOutsideLonLat(int iMM, bool Readtxt, struct MMesh *MM){
int nodemore;
if (Readtxt) {
   nodemore = MM[iMM].firstnodeborder;    // ignore the masked bc creations
printf("AddOutsideLonLat iMM=%d  firstnodeborder=%d old node=%d",iMM,nodemore,MM[iMM].node);
   ifstream file{"LatLon.txt"};
   float VAR[2];
   float value;
   while (file>>value){
       MM[iMM].Lon[nodemore] = value;
       file>>value;
       MM[iMM].Lat[nodemore]=value;
       MM[iMM].depth[nodemore]= 5.;
       //cout << nodemore<< " v="<< value <<" V0="<<VAR[0]<<" V1="<<VAR[1]<< endl;
       nodemore++;      // increment with read. will be num of points not the index
      }

   MM[iMM].node = nodemore;
   printf(" and now MM[%d].node=%d\n",iMM,MM[iMM].node);
   }   // end of add text file points
   else  // just add a box of points on far outside
   {
      // Xbox will be changed to meters in Particle initialize
      MM[iMM].Xbox[0]= -78.;    // Lon min
      MM[iMM].Xbox[1]=  -73.;    //Lon max
      MM[iMM].Xbox[2]= 36.;    // Lat min
      MM[iMM].Xbox[3]=  41.;    // Lat max

      nodemore = MM[iMM].node;

      for (int i=0; i<2; i++){
         for (int j=0; j<2; j++){
            MM[iMM].Lon[nodemore] = MM[iMM].Xbox[i];
            MM[iMM].Lat[nodemore] = MM[iMM].Xbox[j+2];
            MM[iMM].depth[nodemore]= 5.;
            nodemore++;
         }
      }
      MM[iMM].Lon[nodemore] = (MM[iMM].Xbox[0]+MM[iMM].Xbox[1] )/2.;  // 01 12 23 34
      MM[iMM].Lat[nodemore] = MM[iMM].Xbox[2];  // 01 12 23 34
      MM[iMM].depth[nodemore]= 5.;
      nodemore++;
      MM[iMM].Lon[nodemore] = (MM[iMM].Xbox[0]+MM[iMM].Xbox[1] )/2.;  // 01 12 23 34
      MM[iMM].Lat[nodemore] = MM[iMM].Xbox[3];  // 01 12 23 34
      MM[iMM].depth[nodemore]= 5.;
      nodemore++;
      MM[iMM].Lon[nodemore] = MM[iMM].Xbox[0];  // 01 12 23 34
      MM[iMM].Lat[nodemore] = (MM[iMM].Xbox[2]+MM[iMM].Xbox[3] )/2.;  // 01 12 23 34
      MM[iMM].depth[nodemore]= 5.;
      nodemore++;
      MM[iMM].Lon[nodemore] = MM[iMM].Xbox[1];  // 01 12 23 34
      MM[iMM].Lat[nodemore] = (MM[iMM].Xbox[2]+MM[iMM].Xbox[3] )/2.;  // 01 12 23 34
      MM[iMM].depth[nodemore]= 5.;
      nodemore++;

   MM[iMM].node = nodemore;
   }  // end of add box

}




///////////////////////////////////////////////////////
//////////////      FIELDS      ///////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

void ReadMeshField(string& filename, int icase, struct MMesh *MM)
{
   // ReadMeshField will read one of four MM[0:3], U,V,W,Angle,depth

//  icase =   0-Regular, 1-Field U,  2-Field V, 3-Field W, Angle and depth
int iMM=0;
if (icase>0) iMM=icase-1;   //  fill up MM[0] with U, MM[1] with V etc.
NcDim dim;
long ij;
int nx,ny;
int node,nodemore;
// codes for netcdf variables Regular,  
std::vector<std::string> MeshCode={        // mask, x,y, bathy, sigma(or depth)
            "mask","Longitude","Latitude","h","Depth",                         //  bathymetry, sigma levels
            "mask_u","lon_u","lat_u","","s_rho",        // u    eta_u, xi_u
            "mask_v","lon_v","lat_v","","s_rho",        // v    eta_v, xi_v      
            "mask_rho","lon_rho","lat_rho","h","s_w",  // w  eta_rho, xi_rho  h=depth, s_rho=sigma  
            "mask_rho","lon_rho","lat_rho","angle","s_w",     // "angle between XI-axis and EAST" angle and sigma
            "mask_psi","lon_psi","lat_psi",""   //  psi    eta_psi,  xi_psi
        };
cout<<"ReadMeshField: " << filename << "   icase="<<icase<<endl;
NcFile dataFile(filename, NcFile::read);

printf(" icase %d   MeshCode[%d]=",icase,(0+5*icase));
cout << MeshCode[0+5*icase]<<endl;
   NcVar data=dataFile.getVar(MeshCode[0+5*icase]);
   if(data.isNull()) printf(" data.isNull Mask/n");
for (int i=0; i<2; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) ny = dimi;
      if (i==1) nx = dimi;

      cout<<"ReadMeshField  i= "<<i<<" dimi= "<<dimi<<endl;
   }
double Mask[ny][nx];
double LLL[ny][nx];
//double LLat[ny][nx];
//std::vector<std::vector<double> > LLL( ny , std::vector<double> (nx));  
//std::vector<std::vector<double> > Mask( ny , std::vector<double> (nx));  


printf("sizeof LLL  %ld\n",sizeof(LLL));

printf("sizeof Mask %ld\n",sizeof(Mask));

   data.getVar(Mask);
   ij=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
    { MM[iMM].Mask[ij] = Mask[i][j]; ij++; }  }

printf(" icase %d   MeshCode[%d]=",icase,(1+5*icase));
cout << MeshCode[1+5*icase]<<endl;
   data=dataFile.getVar(MeshCode[1+5*icase]);
   if(data.isNull()) printf(" data.isNull Longitude /n");
   data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[iMM].Lon[ij] = LLL[i][j]; ij++;} }  }


 node = ij;
 int summask=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++){
       if((i-1)>=0) {
                        summask += Mask[i-1][j];
         if((j+1)<nx) summask += Mask[i-1][j+1];
         if((j-1)>=0  ) summask += Mask[i-1][j-1];
       }
       if((i+1)<ny) {
                        summask += Mask[i+1][j];
         if((j+1)<nx) summask += Mask[i+1][j+1];
         if((j-1)>=0  ) summask += Mask[i+1][j-1];
       }
       if((j+1)<nx)   summask += Mask[i][j+1];
       if((j-1)>=0  )   summask += Mask[i][j-1];
      
       if (Mask[i][j]==0 && summask > 0 ){
          MM[iMM].Lon[node] = LLL[i][j];
          MM[iMM].depth[node]=0.0;
          node++;}  
          summask=0; 
      } }  // i,j loop

printf(" icase %d   MeshCode[%d]=",icase,(2+5*icase));
cout << MeshCode[2+5*icase]<<endl;
  data=dataFile.getVar(MeshCode[2+5*icase]);
  if(data.isNull()) printf(" data.isNull Latitude or xi_rho/n");
  data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[iMM].Lat[ij] = LLL[i][j]; ij++;} }  }

 node = ij;
 MM[iMM].firstnodeborder=node;  // initialize the first border node
printf(" \nBefore masked border creation firstnodeborder =%d\n",node);

 summask=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++){
       if((i-1)>=0) {
                        summask += Mask[i-1][j];
         if((j+1)<nx) summask += Mask[i-1][j+1];
         if((j-1)>=0  ) summask += Mask[i-1][j-1];
       }
       if((i+1)<ny) {
                        summask += Mask[i+1][j];
         if((j+1)<nx) summask += Mask[i+1][j+1];
         if((j-1)>=0  ) summask += Mask[i+1][j-1];
       }
       if((j+1)<nx)   summask += Mask[i][j+1];
       if((j-1)>=0  )   summask += Mask[i][j-1];
      
       if (Mask[i][j]==0 && summask > 0 ){
          MM[iMM].Lat[node] = LLL[i][j];
          MM[iMM].depth[node]=0.0;
          node++;}  
          summask=0; 
      } }  // i,j loop

printf(" After masked border creation node =%d\n",node);
 node = ij;
 MM[iMM].firstnodeborder=node;  // initialize the first border node
printf(" Roll back border creation node =%d\n",node);

//  icase==3 for rho grid W, angle, depth     Redo for the regular bathy case
   if ( icase==3) {
printf(" icase= %d  ANGLE and depth iMM=%d \n",icase,iMM);

cout << MeshCode[3+5*icase]<<endl;
      data=dataFile.getVar("h");
      if(data.isNull()) printf(" data.isNull h/n");
      data.getVar(LLL);
      ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[iMM].depth[ij] = LLL[i][j]; ij++;} }  }
       //for (int i=0; i<ij; i+=ij/8)
       //     printf("ReadMeshField h MM[%d].depth[%d]=%g\n",iMM,i,MM[iMM].depth[i]);
   
//   angle, depth always for  rho grid icase==3

      data=dataFile.getVar("angle");
      if(data.isNull()) printf(" data.isNull h/n");
      data.getVar(LLL);
      ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[iMM].ANGLE[ij] = LLL[i][j]; ij++;} }  }
       //for (int i=0; i<ij; i+=ij/8)
       //     printf("ReadMeshField angle MM[%d].ANGLE[%d]=%g\n",iMM,i,MM[iMM].ANGLE[i]);
   }
   else     // not case=3, angle and depth set to dummy values 
      { for (int i=0; i<node; i++){
         MM[iMM].ANGLE[i] = 180.;
         MM[iMM].depth[i] =  66.;
         }
      }
 MM[iMM].node = MM[iMM].firstnodeborder;  // delete the bad border points    initialize the first border node


   /* Although labeled sigma, in Regulargrid this is Depth
   Apparently the regular grid gets rid of sigma variations
   and just uses fixed depths for the 3d
   */
  // Sigma   Regular grid, or angle/w rho grid
 // All get a sigma, dimension can change 
printf(" icase %d   MeshCode[%d]=",icase,(4+5*icase));
cout << MeshCode[4+5*icase]<<endl;
      data=dataFile.getVar(MeshCode[4+5*icase]);
      if(data.isNull()) printf(" data.isNull Depth/n");
      dim = data.getDim(0);
      size_t dimi = dim.getSize();
      MM[iMM].nsigma=dimi;
      cout<<" ReadMesh Depth dimi="<<dimi <<endl;
      double sigma[dimi];
      data.getVar(sigma);
      for (int i=0; i<dimi; i++) MM[iMM].sigma[i]=sigma[i];
      


// not needed as LL Mask will be deleted on exit from this routine
//delete [] LLL;
//delete [] Mask;

dataFile.close();
printf(" ReadMesh                  node = %d\n", MM[iMM].node);
bool Readtxt=true;

// roll back all the border points added above use only the file points
if (Readtxt)  MM[iMM].node = MM[iMM].firstnodeborder;  
printf("\n\n before AddOutsideLonLat node = %d    firstnodeborder = %d \n",MM[iMM].node, MM[iMM].firstnodeborder);
AddOutsideLonLat(iMM,Readtxt,MM);
printf(" after AddOutsideLonLat node = %d    firstnodeborder = %d \n\n\n",MM[iMM].node, MM[iMM].firstnodeborder);


}


string NetCDFfiledate(char* filenametemplate,struct MMesh *MM){
   // Same data is in DData[0].ToDay and MMesh[0].ToDay  
   // But they need separate calls to initialize
// create updated filename based on given root name Regular
// and the time seconds ToDay 
// Call from elsewhere after doing this in mainline:
/*

    MM[0].ToDay +=3600;  // for hourly files
    string newername = NetCDFfiledate(MM);
*/
char buffer [125];
int n;

struct tm * tday = gmtime(&MM[0].ToDay);

    //ToDay +=3600;
    tday = gmtime(&MM[0].ToDay);
    //n = sprintf(buffer
    // ,"/home/tom/code/NOSfiles/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"
    // ,tday->tm_year +1900, tday->tm_mon +1, tday->tm_mday,6*int(tday->tm_hour/6),1+(tday->tm_hour%6));
   // n = sprintf(buffer
   //  ,"/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"
   n = sprintf(buffer,filenametemplate
     ,tday->tm_year +1900, tday->tm_mon +1
     ,tday->tm_year +1900, tday->tm_mon +1 
     ,tday->tm_mday,6*int(tday->tm_hour/6),1+(tday->tm_hour%6));

    string newname;
    for (int i=0; i<n; i++) newname=newname+buffer[i]; 

    return newname;
}

string NetCDFfiledate(char* filenametemplate,struct DData *MM){
   // Same data is in DData[0].ToDay and MMesh[0].ToDay  
   // But they need separate calls to initialize
// create updated filename based on given root name Regular
// and the time seconds ToDay 
// Call from elsewhere after doing this in mainline:
/*
int year = 2019, month=5, day=12, hour= 0, minute=5;
tm today = {}; 
today.tm_year =year-1900 ; 
today.tm_mon = month-1;
today.tm_mday = day;
today.tm_hour = hour;
time_t ToDay = mktime(&today);
DD[0].ToDay = ToDay; 

// Then anywhere do:
    MM[0].ToDay +=3600;  // for hourly files
    string newername = NetCDFfiledate(MM);
*/
char buffer [125];
int n;

struct tm * tday = gmtime(&MM[0].ToDay);
// char filetemplate[]="/home/tom/code/NOSfiles/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"

    //ToDay +=3600;
    tday = gmtime(&MM[0].ToDay);
    //n = sprintf(buffer
    // ,"/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"
   n = sprintf(buffer,filenametemplate
     ,tday->tm_year +1900, tday->tm_mon +1
     ,tday->tm_year +1900, tday->tm_mon +1 
     ,tday->tm_mday,6*int(tday->tm_hour/6),1+(tday->tm_hour%6));

    string newname;
    for (int i=0; i<n; i++) newname=newname+buffer[i]; 

    return newname;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
///////// delaunator.hpp  to provide ////////////////
//// delaunator::Delaunator delaunatore(coords); ////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#include <delaunator.hpp>
#include <cstdio>


/////////////////////////////////////////////////////
///////////////// Use triangulation.cpp to add ele, tri_connect
/////////////////     also add calculation of a_frac, b_frac, c_frac
///////////////// MakeMeshEle(iMM,MM);
/////////////////////////////////////////////////////
void MakeMeshEle(int iMM, struct MMesh *MM)
{
   // in main.h   # define NODE  // maximum size of Lon
   int node = MM[iMM].node;

   int error;

   std::vector<double> coords(node*2);
   int i2=0;
   int iv=0;
   for (int i=0; i<node; i++){

      coords.operator[](iv)=double(MM[iMM].X[i]); iv++;
      coords.operator[](iv)=double(MM[iMM].Y[i]); iv++;

      }


    // Delaunator triangulation happens here
    // the constructor delivers the goods in struct d
    //
    cout<<" Just before Delaunator iMM="<<iMM<<endl;
    delaunator::Delaunator delaunatore(coords);

    printf(" delaunatore.triangles.size()=%d\n",delaunatore.triangles.size());
/*    for(std::size_t i = 0; i < 15; i+=3) {
        printf(
            "Triangle points:i=%d [[%f, %f], [%f, %f], [%f, %f]]\n",i,
            delaunatore.coords[2 * delaunatore.triangles[i]],        //tx0
            delaunatore.coords[2 * delaunatore.triangles[i] + 1],    //ty0
            delaunatore.coords[2 * delaunatore.triangles[i + 1]],    //tx1
            delaunatore.coords[2 * delaunatore.triangles[i + 1] + 1],//ty1
            delaunatore.coords[2 * delaunatore.triangles[i + 2]],    //tx2
            delaunatore.coords[2 * delaunatore.triangles[i + 2] + 1] //ty2
        );
        printf("node# of Triangle i=[%d]   [  %d  %d  %d ]\n",i,
            delaunatore.triangles[i],delaunatore.triangles[i+1],delaunatore.triangles[i+2] );
        printf("Triangle halfedges i=[%d]   [  %d  %d  %d ]\n",i,
            delaunatore.halfedges[i],delaunatore.halfedges[i+1],delaunatore.halfedges[i+2] );
        printf("Edges of Triangle  i=[%d]   [  %d  %d  %d ]\n",i,
            3*i,3*i+1, 3*i+2 );
    }
*/
    int nele;
    for(std::size_t i = 0; i < delaunatore.triangles.size(); i+=3) {
        nele=i/3;
        MM[iMM].ele[nele][0]= delaunatore.triangles[i];
        MM[iMM].ele[nele][1]= delaunatore.triangles[i+1];
        MM[iMM].ele[nele][2]= delaunatore.triangles[i+2];
        MM[iMM].tri_connect[nele][0] = (delaunatore.halfedges[i+1] );
        MM[iMM].tri_connect[nele][1] = (delaunatore.halfedges[i+2] ); 
        MM[iMM].tri_connect[nele][2] = (delaunatore.halfedges[i+0] ); 
        if (MM[iMM].tri_connect[nele][0]>0) MM[iMM].tri_connect[nele][0]  /= 3;
        if (MM[iMM].tri_connect[nele][1]>0) MM[iMM].tri_connect[nele][1]  /= 3; 
        if (MM[iMM].tri_connect[nele][2]>0) MM[iMM].tri_connect[nele][2]  /= 3; 
    }
    MM[iMM].nele = nele;





// Calculate a_frac, b_frac, c_frac for each ele, ie triangle
  float x3[3], y3[3], xo ,yo;
  float d;
//printf("start ele_func\n");

  for (int i=0; i<nele; i++) {  
   	for (int k=0;k<3;k++) {
      		x3[k] = MM[iMM].X[MM[iMM].ele[i][k]] ;
      		y3[k] = MM[iMM].Y[MM[iMM].ele[i][k]] ;
   	}

	// Determinate [x3 y3 ones(3,1)] 
  	d= x3[0]*y3[1] +x3[1]*y3[2] +x3[2]*y3[0] -x3[0]*y3[2] -x3[1]*y3[0] -x3[2]*y3[1] ; 
	
	//  Calculate a,b,c for every triangle (3 k node's / triangle)
    	for(int k=0; k<3; k++) {
         MM[iMM].a_frac[i][k] = (y3[1]-y3[2])/d;
        	MM[iMM].b_frac[i][k] = (x3[2]-x3[1])/d;
        	MM[iMM].c_frac[i][k] = (x3[1]*y3[2]-x3[2]*y3[1])/d;
		//   rotate the x,y for next function on node k
	 	xo=x3[0]; 
		x3[0]=x3[1]; 
		x3[1]=x3[2]; 
		x3[2]=xo;
	 	yo=y3[0]; 
		y3[0]=y3[1]; 
		y3[1]=y3[2]; 
		y3[2]=yo;
    	} 
  }
  printf("MakeMeshEle finished ele_func\n");


} 