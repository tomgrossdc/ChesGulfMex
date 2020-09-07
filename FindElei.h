// FindElei.h   
// subrountine FindElei(Lon,Lat,MM) to be included in main.h
// used to find the element number of closest node to Lon, Lat

int FindElei(float Lon,float Lat, struct MMesh *MM, int iMM)
{
    //int iMM=2;
    int inode=0;
    float minLL = 50.*50.;   // dist is lon, lat degrees squared 
    for (int i=0; i<MM[iMM].node; i++)
    {
        float disttest = (MM[iMM].Lon[i]-Lon)*(MM[iMM].Lon[i]-Lon)
                     + (MM[iMM].Lat[i]-Lat)*(MM[iMM].Lat[i]-Lat);
        if (disttest<minLL) { inode=i; minLL=disttest; }
    }
//    printf("\n FindElei Lon=%g Lat=%g inode=%d MM[%d].Lon=%g MM[%d].Lat=%g /n",
//        Lon, Lat, inode, iMM, MM[iMM].Lon[inode],iMM, MM[iMM].Lat[inode]);
    return inode;
}      