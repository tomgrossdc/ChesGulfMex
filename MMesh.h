//test .
#ifndef MESH_H
#define MESH_H

#include "Main.h"
#include "string.h"

void ReadMesh(string& filename, struct MMesh *MM);
void AddOutsideLonLat(int iMM, bool Readtxt, struct MMesh *MM);
void AddOutsideLonLatG(int iMM, bool Readtxt, struct MMesh *MM);

void MakeMeshEle(int iMM, struct MMesh *MM);

void BuildMM(struct MMesh *MM, struct CControl CC);


#endif
