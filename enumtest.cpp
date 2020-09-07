// simple enum testing
// g++ enumtest.cpp

#include <iostream>
#include "string.h"
#include <fstream>

using namespace std;



struct CControl
{
   // mainpart.cu

   int NUM_PARTICLEs; //  these used #define;  to be changed
   int NODe;          // and were set as external globals
   int NSIGMa;

   char* filetemplate;
   string shadervs;
   string shaderfs;

   
   ;                // for reading NetCDF data files
   int year = 2020; // starting date for NetCDF files
   int month = 02;
   int day = 10;
   int hour = 12;

   // particle.cpp
   enum class Layouts
   {
      BoxRows,
      FillBay,
      BorderBay,
      SewerPipe,
      Oysters
   };
   Layouts layout = Layouts::SewerPipe;
   int numrows = 100;
   float Pipell[2] = {-76.617, 38.162}; // -77.03, 38.75 DC; -76.617, 38.162 mid Potomac; -76.48, 39.175 Baltimore;
   int iage_step = 2;                   // dt_sec between releases of particles
   float rand_spread = 1000.;

   bool read_control_data()
   {
      string varname;
      string value;
      ifstream file{"CControl_Data.txt"};
      while (file >> varname)
      {
         //file >> value;
         cout << varname << endl;

         if (varname== "NUM_PARTICLEs")
            file >> NUM_PARTICLEs;
         else if (varname== "NODe")
            file >> NODe;
         else if (varname=="NSIGMa")
            file >> NSIGMa;
         else if (varname=="filetemplate") {
            file>>filetemplate;
         }
         else if (varname=="year")
            file >> year;
          else if (varname=="month")
            file >> month;
         else if (varname=="day")
            file >> day;
         else if (varname=="hour")
            file >> hour;
         else if (varname=="numrows")
            file >> numrows;
         else if (varname=="Pipell[0]")
            file >> Pipell[0];
         else if (varname=="Pipell[1]")
            file >> Pipell[1];
         else if (varname=="iage_step")
            file >> iage_step;
         else if (varname=="rand_spread")
            file >> rand_spread;
           
         else if(varname=="Layout"){
            file >> value;
            if (value=="BoxRows") layout = Layouts::BoxRows;
            else if (value=="SewerPipe") layout = Layouts::SewerPipe ;
            else if (value=="FillBay") layout = Layouts::FillBay ;
            else if (value=="BorderBay") layout = Layouts::BorderBay ;
            else if (value=="Oysters") layout = Layouts::Oysters ;
            else {cout << "Invalid CControl_Data.txt Layout : " << value << endl;
            return false;
            }
         }
            
         else
            {cout << "Invalid CControl_Data.txt varname : " << varname << endl;
            return false;
            }
      }
      return true;
   }
};
void PPartInit( struct CControl *CC){
switch(CC->layout){
    case CC->Layouts::BoxRows:{
      CC->shadervs="shaderbox.vs"; 
      CC->shaderfs="shaderbox.fs"; 
    } break;

    case CC->Layouts::SewerPipe:{
      CC->shadervs="shaderpipe.vs"; 
      CC->shaderfs="shaderpipe.fs"; 
    } break;

    case CC->Layouts::FillBay:{
      CC->shadervs="shaderfill.vs";
      CC->shaderfs="shaderfill.fs";      
    } break;

    case CC->Layouts::BorderBay:{
      CC->shadervs="shaderborder.vs"; 
      CC->shaderfs="shaderborder.fs";        
    } break;

    default: { printf("error particle case");}
}  // switch closure

    cout <<" Particle PPartInit CC->shadervs = "<< CC->shadervs << endl;
    cout <<" Particle PPartInit CC->shaderfs = "<< CC->shaderfs << endl;

}

void GetVS(string shadername){
   cout << "GetVS shadername="<<shadername<<endl;
   ifstream file{shadername};
   string varname;
   while (file >>varname) cout <<varname<<endl;
}

extern struct CControl CC;

CControl CC;

int main()
{
   char st[128] = "Stuff in string";
   CC.filetemplate = st;
   cout << CC.filetemplate << endl;
   bool error = CC.read_control_data();
   cout << "error =" << error << endl;

cout<< CC.NUM_PARTICLEs<<endl;
cout << CC.NODe <<endl;
cout << CC.NSIGMa <<endl;
cout << CC.year <<endl;
cout << CC.month <<endl;
cout << CC.day <<endl;
cout << CC.hour <<endl;
cout << CC.numrows <<endl;
cout << CC.Pipell[0] <<endl;
cout << CC.Pipell[1] <<endl;
cout << CC.iage_step <<endl;
cout << "filetemplate= "<<CC.filetemplate <<endl;

PPartInit( &CC);
cout << "shadervs= "<<CC.shadervs <<endl;
cout << "shaderfs= "<<CC.shaderfs <<endl;

GetVS(CC.shadervs);

cout << " end of programme" <<endl;

}
