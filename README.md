# CudaParticleMover
Particle tracker using Nvidia Cuda cores and NetCDF files from NOAA/NOS

Sept 2020 updates include support for FVCOM as well as ROMS.  Example input files
are in directory CControl.  

Building the code:
Use a computer with an Nvidia graphics card. 
Install Netcdf libraries
Install Nvidia libraries
Tailor Makefile to use those libraries
Tailor Cronjob.sh to download NetCDF files and write them to your own directory system. 

April 2020 
New version is mainly complete.  CControl_Data.txt holds many adjustable parameters to control the runs.
Notes in the file give examples of how to use those parameters.

Next task will be to add active fall velocity to the PP particles. 

