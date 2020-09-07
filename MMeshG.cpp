/*   Alter this by eliminating netcdf from set_Mesh 
    include reading a simple text file for lon, lat, depth, sigma
*/


#include "MMesh.h"






///////////////////////////////////////////////////////
//////////////      FIELDS      ///////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

//NGOFS FVCOM
void ReadMeshFieldG(string& filename, int icase, struct MMesh *MM)
{
   // ReadMeshFieldG will read one of four MM[0:3], U,V,W,Angle,depth

//  icase =   0-Regular, 1-Field U,  2-Field V, 3-Field W, Angle and depth
int iMM=0;
if (icase>0) iMM=icase-1;   //  fill up MM[0] with U, MM[1] with V etc.
NcDim dim;
long ij;
int nx,ny;
int node,nodemore;
// codes for netcdf variables 
// u lonc, latc,  MM[0]
// v lonc, latc,  MM[1]
// ww lonc, latc,  MM[2]
// h lon, lat,     MM[3]      h, zeta temp, salinity

cout<<"ReadMeshFieldG: " << filename << "   icase="<<icase<<endl;
NcFile dataFile(filename, NcFile::read);

NcVar data;
int xnode;
//for (iMM=0;iMM<4; iMM++){
if (iMM==3) data=dataFile.getVar("lon");
else        data=dataFile.getVar("lonc");
      dim = data.getDim(0);
      node = dim.getSize();
MM[iMM].firstnodeborder=node;
MM[iMM].node=node;
      printf(" lon[%d]\n",node);
      data.getVar(MM[iMM].Lon);
      xnode = 10;
      printf("MM[%d].Lon[%d]=%g\n",iMM,xnode,MM[iMM].Lon[xnode]);
      xnode = node-1;
      printf("MM[%d].Lon[%d]=%g\n",iMM,xnode,MM[iMM].Lon[xnode]);

if (iMM==3) data=dataFile.getVar("lat");
else        data=dataFile.getVar("latc");
      dim = data.getDim(0);
      node = dim.getSize();
      printf(" lat[%d]\n",node);
      data.getVar(MM[iMM].Lat);
      xnode = 10;
      printf("MM[%d].Lat[%d]=%g\n",iMM,xnode,MM[iMM].Lat[xnode]);
      xnode = node-1;
      printf("MM[%d].Lat[%d]=%g\n",iMM,xnode,MM[iMM].Lat[xnode]);


if (iMM==3) 
{  data=dataFile.getVar("h");
   data.getVar(MM[iMM].depth);
}
else
{
   // if a latc type variable then build depth of nele
// first get nv the element pointer array
   data=dataFile.getVar("nv");
   dim = data.getDim(0);
   int three = dim.getSize();
   dim = data.getDim(1);
   int nele = dim.getSize();
   int nv[three][nele];
   data.getVar(nv);
   float Depth[node];
   data=dataFile.getVar("h");
   data.getVar(Depth);
   for (int i=0; i<nele; i++){
      MM[iMM].depth[i] = (Depth[nv[0][i]]+ Depth[nv[1][i]]+Depth[nv[2][i]])/3.0;
   }

   for (int ipp = 500; ipp < 5000; ipp+=500)
   printf(" MM[%d].nv[0-2][%d]= %d %d %d    %f %f %f\n",iMM,ipp,nv[0][ipp],nv[1][ipp],nv[2][ipp],
   Depth[nv[0][ipp]], Depth[nv[1][ipp]],Depth[nv[2][ipp]]  );

}



      //for (int i=0; i<node; i+=1){
      //   if (iMM==0 && i<85720 && i>85700) printf("MMMM[iMM].depth[%d]=%f\n",i,MM[iMM].depth[i]);
      //}




      for (int i=0; i<node; i++){
         MM[iMM].ANGLE[i] = 0.;
         }

      data=dataFile.getVar("siglay");
      dim = data.getDim(0);
      MM[iMM].nsigma = dim.getSize();
      printf(" MM[%d].nsigma=%d\n",iMM,MM[iMM].nsigma);
      /*   Sigma is not working
float siglay[400000];   // Does not work when =40*100000 
      dim = data.getDim(1);
      printf("siglaynode=%d\n",dim.getSize());
      //data.getVar(siglay);
      for (int i=500; i<700; i+=335)
         { for(int j=0; j<MM[iMM].nsigma; j++)
              { printf("%g[%d][%d]=%g, ",i,j,siglay[i+j]); }
         printf("/n");
         }
      */
      for (int i=0; i<MM[iMM].nsigma; i++){
         //  upside down first  MM[iMM].sigma[i] = -float(i)/float(MM[iMM].nsigma -1);
         MM[iMM].sigma[i] =  -1.+float(i)/float(MM[iMM].nsigma -1);
         printf(" %g, ",MM[iMM].sigma[i]);
         }
      printf("\n");
//}

//iMM=0;

dataFile.close();
printf(" ReadMeshFieldG                 node = %d\n", MM[iMM].node);
bool Readtxt=false;

// roll back all the border points added above use only the file points
if (Readtxt)  MM[iMM].node = MM[iMM].firstnodeborder;  
printf("\n\n before A ddOutsideLonLat node = %d    firstnodeborder = %d \n",MM[iMM].node, MM[iMM].firstnodeborder);
AddOutsideLonLatG(iMM,Readtxt,MM);
printf(" after A ddOutsideLonLat node = %d    firstnodeborder = %d \n\n\n",MM[iMM].node, MM[iMM].firstnodeborder);


}

void AddOutsideLonLatG(int iMM, bool Readtxt, struct MMesh *MM){
int nodemore;
if (Readtxt) {
   nodemore = MM[iMM].firstnodeborder;    // ignore the masked bc creations
printf("A ddOutsideLonLatG iMM=%d  firstnodeborder=%d old node=%d",iMM,nodemore,MM[iMM].node);
   ifstream file{"LonLatNWGOFS.txt"};
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
      MM[iMM].Xbox[0]= 262.;    // Lon min
      MM[iMM].Xbox[1]= 276.;    //Lon max
      MM[iMM].Xbox[2]= 25.;    // Lat min
      MM[iMM].Xbox[3]=  31.;    // Lat max

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




string NetCDFfiledateG(char* filenametemplate,struct MMesh *MM){
//  Special case for GulfMex files which use t03z 9 15 21 instead of 0 6 12 18 +6
//  So rewrite the crontab that captures the files and will change to simpler file names
// nos.ngofs.fileds.20200701.n000.nc     n000 n001 n002 ... n022 n023
//  PS, the original filename is based on Eastern Time, not UTC. 
//  A kludge workaround was to add 3 hours so that the day would align for 24 file captures
//  The internal netcdf time is not changed, but the starting file might be 3 hours off.
//   Add a conditional 3 hour offset to the CControl input hour
//
// Same data is in DData[0].ToDay and MMesh[0].ToDay  
// But they need separate calls to initialize
// create updated filename based on given root name Regular
// and the time seconds ToDay 
// Call from elsewhere after doing this in mainline:
/*

    MM[0].ToDay +=3600;  // for hourly files
    string newername = NetCDF filedate(MM);
*/
char buffer [125];
int n;

struct tm * tday = gmtime(&MM[0].ToDay);

    //ToDay +=3600;
    tday = gmtime(&MM[0].ToDay);

   // cbofs:
   // /media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"
   //n = sprintf(buffer,filenametemplate
   //  ,tday->tm_year +1900, tday->tm_mon +1
   //  ,tday->tm_year +1900, tday->tm_mon +1 
   //  ,tday->tm_mday,6*int(tday->tm_hour/6),1+(tday->tm_hour%6));

   // ngofs:
   // /media/tom/A8TLinux/NOSnetcdf/NGOFS/202007/nos.ngofs.fields.20200706.n023.nc
   ///media/tom/A8TLinux/NOSnetcdf/NGOFS/%d%02d /nos.ngofs.fields.%d%02d%02d.n%03d.nc
   n = sprintf(buffer,filenametemplate
     ,tday->tm_year +1900, tday->tm_mon +1
     ,tday->tm_year +1900, tday->tm_mon +1 
     ,tday->tm_mday,tday->tm_hour);

    string newname;
    for (int i=0; i<n; i++) newname=newname+buffer[i]; 

cout<< " Inside NETCDFfiledate(MM)"<< newname <<endl;

    return newname;
}
string NetCDFfiledateG(char* filenametemplate,struct DData *MM){
//  Special case for GulfMex files which use t03z 9 15 21 instead of 0 6 12 18 +6
// Extra to allow DData to work

char buffer [125];
int n;

struct tm * tday = gmtime(&MM[0].ToDay);

    //ToDay +=3600;
    tday = gmtime(&MM[0].ToDay);

   n = sprintf(buffer,filenametemplate
     ,tday->tm_year +1900, tday->tm_mon +1
     ,tday->tm_year +1900, tday->tm_mon +1 
     ,tday->tm_mday,tday->tm_hour);

    string newname;
    for (int i=0; i<n; i++) newname=newname+buffer[i]; 

cout<< " Inside NETCDFfiledate(DD)"<< newname <<endl;

    return newname;
}



