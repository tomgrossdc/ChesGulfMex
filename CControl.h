// CControl.h   
// struct CControl to be included in main.h
#define DT_SEC 4.5                // 3.0 very smooth but ponderous

struct CControl {
    // mainpart.cu 
    int NUM_PARTICLEs;    //  these used #define;  to be changed
    int NODe;     // and were set as external globals
    int NSIGMa;

  // for reading NetCDF data files
    char* filetemplate;
    int year = 2020;     // starting date for NetCDF files
    int month=01;
    int day=01;
    int hour= 06;

    // particle.cpp
    enum class Layouts {AreaFill, FillBay, BorderBay, Pipe_Sources, Oysters };
    Layouts layout = Layouts::Pipe_Sources;
    string run_mode = "Pipe_Sources";
    enum class ColorModes {ColorByRelease, ColorByAge, ColorByPulse, ColorByDepth, ColorByOrigin, ColorByTide, ColorBySigma };
    ColorModes Colortype = ColorModes::ColorByAge;
    int color_mode=0;  //default
    int numrows=200;
    float Pipell[20];    // Stored [lat0 lon0 lat1 lon1, ] .. -77.03, 38.75 DC; -76.617, 38.162 mid Potomac; -76.48, 39.175 Baltimore;  
    int NumPipes;
    int iage_step = 2;   // dt_sec between releases of particles
    float ReleaseIncrement = 2.0;  //  float version of iage_step
    float pulse_duration = 1.0;
    float pulse_spacing = 0.0; 
    float age_class = 7200.0; 
    float rand_spread = 1000.;
    float depthcolorinterval=5.;
    int Dot_Size=1;
   string shadervs="Shaders/shaderRainbow.vs";
   string shaderfs="Shaders/shader.fs";
   float WfCore=0.0;      // Wf=WfCore*cos(time_now*WfFreq - WfShft);
   float WfFreq=0.0;
   float WfShft=0.0;
   float KH=0.0;
   float KV=0.0;
   float LONmid = 265.400;
   float LATmid = 29.20;
   float LONwidth=1.5;
   float LAThieght = .8;
   bool IsFVCOM=false;
   int isfvcom = 0;

 bool read_control_data(string inputfilename)
   {
      string varname;
      string value;
      ifstream file{inputfilename};
      NumPipes=0;
      while (file >> varname)
      {
         //file >> value;
         cout << varname << endl;

         if (varname== "NUM_PARTICLEs")
            file >> NUM_PARTICLEs;
         else if (varname== "NODe")
            file >> NODe;
         else if (varname== "LONmid")
            file >> LONmid;
         else if (varname== "LONwidth")
            file >> LONwidth;
         else if (varname== "LATmid")
            file >> LATmid;
         else if (varname== "LAThieght")
            file >> LAThieght;
         else if (varname=="NSIGMa")
            file >> NSIGMa;
         else if (varname=="filetemplate") 
            file>>filetemplate;
         else if (varname=="isfvcom")
            {file>>isfvcom;
            if (isfvcom==1) IsFVCOM=true;}
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
         else if (varname=="Pipell")
            {file >> Pipell[0+NumPipes];
            file >> Pipell[1+NumPipes];
            NumPipes+=2; }
         else if (varname=="iage_step")
            file >> iage_step;
         else if (varname=="ReleaseIncrement")
            file >> ReleaseIncrement;
         else if (varname=="pulse_duration")
            { file >> pulse_duration; 
              if (pulse_duration<1.0) pulse_duration=1.0;}
         else if (varname=="pulse_spacing")
            file >> pulse_spacing;
         else if (varname=="age_class")
            file >> age_class;
         else if (varname=="rand_spread")
            file >> rand_spread;
         else if (varname=="Dot_Size")
            file >> Dot_Size;
         else if (varname=="depthcolorinterval")
            file>> depthcolorinterval;
         else if (varname=="WfCore")
            file>> WfCore;
         else if (varname=="WfFreq")
            file>> WfFreq;
         else if (varname=="WfShft")
            file>> WfShft;
         else if (varname=="KH")
            {file>> KH;  KH = KH*sqrt(DT_SEC);}
         else if (varname=="KV")
            {file>> KV;  KV = KV*sqrt(DT_SEC);}       

         else if(varname=="Layout"){
            file >> value;
            run_mode=value;
            cout<<"varname="<<varname<<"   value="<<value<<endl;
            if (value=="AreaFill") layout = Layouts::AreaFill;
            else if (value=="Pipe_Sources") layout = Layouts::Pipe_Sources ;
            else if (value=="FillBay") layout = Layouts::FillBay ;
            else if (value=="BorderBay") layout = Layouts::BorderBay ;
            else if (value=="Oysters") layout = Layouts::Oysters ;
            else {cout << "Invalid CControl_Data.txt Layout : " << value << endl;
            return false;
            } }

         else if(varname=="Colortype"){
            file >> value;
            cout<<"Colortype="<<varname<<"   value="<<value<<endl;
            if       (value=="ColorByRelease")
               {Colortype = ColorModes::ColorByRelease;     color_mode= 0;
               shadervs="Shaders/shaderRainbow.vs";
               shaderfs="Shaders/shader.fs";}
            else if  (value=="ColorByAge") 
               {Colortype = ColorModes::ColorByAge;     color_mode= 1;
               shadervs="Shaders/shaderRedBlue.vs";
               shaderfs="Shaders/shader.fs";}
            else if (value=="ColorByPulse") 
               {Colortype = ColorModes::ColorByPulse;color_mode= 2;
               shadervs="Shaders/shaderRedBlue.vs";
               shaderfs="Shaders/shader.fs";}
            else if (value=="ColorByDepth") 
               {Colortype = ColorModes::ColorByDepth;   color_mode= 3;
               shadervs="Shaders/shaderRedRed.vs";
               shaderfs="Shaders/shader.fs";}
            else if (value=="ColorByOrigin") 
               {Colortype = ColorModes::ColorByOrigin;   color_mode= 4;
               shadervs="Shaders/shaderRainbow.vs";
               shaderfs="Shaders/shader.fs";}
            else if (value=="ColorByTidal") 
               {Colortype = ColorModes::ColorByDepth;   color_mode= 3;
               shadervs="Shaders/shadertidal.vs";
               shaderfs="Shaders/shader.fs";}
            else if (value=="ColorBySigma") 
               {Colortype = ColorModes::ColorBySigma;   color_mode= 5;
               shadervs="Shaders/shaderSigma.vs";
               shaderfs="Shaders/shader.fs";}
            else {cout << "Invalid CControl_Data.txt Colortype : " << value << endl;
            return false;
            } }

            
         else
            {cout << "Invalid CControl_Data.txt varname (or begin comments) : " << varname << endl;
            NumPipes=NumPipes/2;   // counter lats + lons  / 2 
            return false;
            }
      }
      return true;
   }
};

extern struct CControl CC;
