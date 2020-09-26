// data.h
#ifndef DATA_H
#define DATA_H

#include "Main.h"

#include <iostream>
#include <netcdf>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;


void BuildDD(struct DData *DD, struct MMesh *MM, struct CControl CC);

void ReadData(double time_now, int ifour, struct DData *DD, struct MMesh *MM);

void ReadDataRegNetCDF(string& filename, int ifour, struct DData *DD, struct MMesh *MM);
void ReadFieldNetCDF(string& filename, int ifour, struct DData *DD, struct MMesh *MM);
void ReadFieldNetCDFG(string& filename, int ifour, struct DData *DD, struct MMesh *MM);

#endif
