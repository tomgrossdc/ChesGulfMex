#include "PParticle.h"

// extern CControl CC;

// Initialize PP* by declaring and malloc'ing in mainline before calling this:
void PPartInit(struct PPart *host_P, struct MMesh *MM, 
           struct CControl *CC, int num_P)
{
//    CControl CC;

        float DEG_PER_METER= 90./(10000*1000); 

///enum class Layouts {BoxRows, FillBay, BorderBay, Pipe_Sources };
//Layouts layout = Layouts::FillBay;
//Layouts layout = Layouts::Pipe_Sources;

//float rand_spread = 1000.;
//case   box
        printf("\n Start PPartInit %d  num_P = %d\n",CC->layout, num_P );

switch(CC->layout){
    case CC->Layouts::AreaFill:{
        printf(" Start AreaFill \n");
        // Even Spaced FillBay using areas of elements
// First Fill the white border:
        int ipcount = 0;
        for (int ip=MM[0].firstnodeborder; ip<MM[0].node; ip++) {
            // White border points
            host_P[ipcount].x_present = MM[0].X[ip] ;  
            host_P[ipcount].y_present = MM[0].Y[ip] ;  
            host_P[ipcount].z_present = 0.0;    //MM[0].depth[imesh]/2.;
            host_P[ipcount].state = 0;  // white boundary, never move
            host_P[ipcount].age = -57600*60;

            host_P[ipcount].num_P = num_P;
            host_P[ipcount].p_id = ip;
            host_P[ipcount].i_ele = 55;
            ipcount++; 
        }
        printf("AreaFill  ipcount = %d   MM.nele = %d    num_P = %d\n",ipcount,MM[0].nele,num_P);
        // Find area of good elements

        //float Xpipe = (( CC->Pipell[0]-CC->LONmid) /DEG_PER_METER )*cos(CC->LATmid * PI/180.);
  	    //float Ypipe =  ( CC->Pipell[1]-CC->LONmid) /DEG_PER_METER;


        // Loop on elements
        int nele = MM[0].nele;
        //int nele = 5000;
        float AreaBay=0.0;
        for (int iele=0; iele<nele; iele++){
            if (MM[0].goodele[iele]){
                float X0=MM[0].X[MM[0].ele[iele][0]];
                float X1=MM[0].X[MM[0].ele[iele][1]];
                float X2=MM[0].X[MM[0].ele[iele][2]];
                float Y0=MM[0].Y[MM[0].ele[iele][0]];
                float Y1=MM[0].Y[MM[0].ele[iele][1]];
                float Y2=MM[0].Y[MM[0].ele[iele][2]];
                AreaBay += abs(X0*(Y1 -Y2)+X1*(Y2 -Y0)+X2*(Y0 -Y1))/2.;
            }
        }
        printf(" AreaBay = %g m2   %g km2 \n",AreaBay , AreaBay/(1000.*1000.));
        // Fill elements with N = num_P*areaelement/areabay
        // Loop on elements
        // Fill elements with N = num_P*areaelement/areabay
        int ip=ipcount;
        for (int iele=0; iele<nele; iele++){
            if (MM[0].goodele[iele]){
                float X0=MM[0].X[MM[0].ele[iele][0]];
                float X1=MM[0].X[MM[0].ele[iele][1]];
                float X2=MM[0].X[MM[0].ele[iele][2]];
                float Y0=MM[0].Y[MM[0].ele[iele][0]];
                float Y1=MM[0].Y[MM[0].ele[iele][1]];
                float Y2=MM[0].Y[MM[0].ele[iele][2]];
                float AreaTri= abs(X0*(Y1 -Y2)+X1*(Y2 -Y0)+X2*(Y0 -Y1))/2.;
                float RadTri= sqrt(abs(AreaTri))/1.;
                float MX = (X0+X1+X2)/3.;
                float MY = (Y0+Y1+Y2)/3.;
                int NPpT= int(num_P*AreaTri/AreaBay);
                 if (iele%37==0) printf(" iele=%d  NPpT = %d \n",iele, NPpT);
                //NPpT = 10;
                NPpT = num_P/nele;
                for (int i=0; i<NPpT; i++){
                    float randangle = 6.28318527*rand()/(float)RAND_MAX;
                    float cr=cos(randangle);
                    float sr=sin(randangle);
                    float radius = RadTri*(1.-pow(rand()/(float)RAND_MAX,2)) ; 
                    //radius = RadTri*.66;
                    host_P[ip].XYZstart[0]= MX + radius*cr;
                    host_P[ip].XYZstart[1]= MY + radius*sr;
                    host_P[ip].XYZstart[2]=  -2.0;
                    if(i==0) {     //  Put one particle just above surface 
                        host_P[ip].XYZstart[0]= MX ;
                        host_P[ip].XYZstart[1]= MY ;
                        host_P[ip].XYZstart[2]=  2.0;}
                //    printf(" P[%d].X[02] %g %g %g \n",ip,host_P[ip].XYZstart[0],
                //              host_P[ip].XYZstart[1],host_P[ip].XYZstart[2]);

                    host_P[ip].state = 3;  // waiting for release time 
                    host_P[ip].age = 1;  // already launched
                    host_P[ip].Release_time = MM[0].time_init + CC->ReleaseIncrement*ip;  // all
                    host_P[ip].p_id = ip;
                    host_P[ip].i_ele = 55;

                    if(ip<num_P) ip++;
                }
            }
        }


        } break;
    case CC->Layouts::Pipe_Sources:{
        //CC->shadervs="shaderpipe.vs"; 
        //CC->shaderfs="shaderpipe.fs"; 
        //case  Pipe_Sources,  Set white border. Then set PPart.age to sequential index
        printf(" layout=Pipe_Sources NumPipes = %d \n",CC->NumPipes);
        int ipcount = 0;
        for (int ip=MM[0].firstnodeborder; ip<MM[0].node; ip++) {
            // White border points
            host_P[ipcount].x_present = MM[0].X[ip] ;  
            host_P[ipcount].y_present = MM[0].Y[ip] ;  
            host_P[ipcount].z_present = 0.0;    //MM[0].depth[imesh]/2.;
            host_P[ipcount].state = 0;  // white boundary, never move
            host_P[ipcount].age = -57600*60;

            host_P[ipcount].num_P = num_P;
            host_P[ipcount].p_id = ip;
            host_P[ipcount].i_ele = 55;
            ipcount++; 
        }

        // Sewer pipe location:
        //float Pipell[]{-76.617, 38.162};    // -77.03, 38.75 DC; -76.617, 38.162 mid Potomac; -76.48, 39.175 Baltimore;  
        float Xpipe[10];
        float Ypipe[10];
          for (int ipipe=0; ipipe<CC->NumPipes; ipipe++){
            int ipp=ipipe*2;
            Xpipe[ipipe] = (( CC->Pipell[ipp]-CC->LONmid) /DEG_PER_METER )*cos(CC->LATmid * PI/180.);
  	        ipp +=1;
            Ypipe[ipipe] =  ( CC->Pipell[ipp]-CC->LATmid) /DEG_PER_METER;
            //Xpipe[ipipe]= (( CC->Pipell[2*ipipe]-LON_LAT_KEY_00) /DEG_PER_METER )*cos(LON_LAT_KEY_01 * PI/180.);
  	        //Ypipe[ipipe] =  ( CC->Pipell[2*ipipe+1]-LON_LAT_KEY_01) /DEG_PER_METER;}
            }
          int iage = 0;
            //int iage_step = 2; 
          int ip=ipcount; 
          while (ip<(num_P)) {
            //iage++;                 // increment by DT_SEC
            iage += CC->iage_step;
            float randangle = 6.28318527*rand()/(float)RAND_MAX;
            float cr=cos(randangle);
            float sr=sin(randangle);
            float radius = CC->rand_spread*(1.-pow(rand()/(float)RAND_MAX,2)) ;
            //host_P[ip].XYZstart[0]= Xpipe + CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);
            //host_P[ip].XYZstart[1]= Ypipe + CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);
          for (int ipipe=0; ipipe<CC->NumPipes; ipipe++){
                if(ip==num_P) ip-=1;  // regardless of number of pipes, ip<num_P
                host_P[ip].XYZstart[0]= Xpipe[ipipe] + radius*cr;
                host_P[ip].XYZstart[1]= Ypipe[ipipe] + radius*sr;
                host_P[ip].XYZstart[2]=  -2.;

                host_P[ip].state = 3;  // waiting for release time 
                host_P[ip].age = -iage;  // release countdown timer
            //    host_P[ip].Release_time = MM[0].time_init + ip*CC->ReleaseIncrement;
            // batch release experiment:  
                host_P[ip].Release_time = MM[0].time_init + ip*CC->ReleaseIncrement
                + floor(CC->ReleaseIncrement*ip/(CC->pulse_duration)) *(CC->pulse_spacing);
printf("%6d %d %g %g \n",ip,ipipe,host_P[ip].XYZstart[0],host_P[ip].XYZstart[1]);
                host_P[ip].num_P = num_P;
                host_P[ip].p_id = ip;
                host_P[ip].i_ele = 55;
                host_P[ip].WfCore=CC->WfCore;      // Wf=WfCore*cos(time_now*WfFreq - WfShft);
                host_P[ip].WfFreq=CC->WfFreq;      //  2*pi/(Period sec)
                host_P[ip].WfShft=CC->WfShft;
                //if(ip%2==0) host_P[ip].WfCore=0.0;
                ip++; 
           }
          }
        //  } huh
        printf("PPartInit  MM[0].firstnodeborder = %d   MM[0].nodes=%d \n", MM[0].firstnodeborder,MM[0].node);

    } break;
    case CC->Layouts::FillBay:{
        //CC->shadervs="shaderfill.vs";
        //CC->shaderfs="shaderfill.fs";      
        //CC->shadervs="shaderdepth.vs";
        //CC->shaderfs="shaderpipe.fs";      
       //case  fill bay with a jiggle at nodal points
        printf("PPartInit  MM[0].firstnodeborder = %d   MM[0].nodes=%d \n", MM[0].firstnodeborder,MM[0].node);
        //MM[0].firstnodeborder=99827;
        int long  imesh =MM[0].firstnodeborder/2;     //  99827  108049;
        imesh = 140328; 
        float lonp=CC->LONmid+CC->LONwidth/2.;
        float lonm=CC->LONmid-CC->LONwidth/2.;
        float latp=CC->LATmid+CC->LAThieght/2.;
        float latm=CC->LATmid-CC->LAThieght/2.;
        printf(" lonm lonp %g %g  latm latp %g %g \n",lonm,lonp,latm,latp);
        for (int ip=0; ip<num_P; ip++) {

            // find a imesh within the bounds
            bool imeshtestcontinue=true;
            while (imeshtestcontinue) {
                //imesh+=217; imesh = imesh%MM[0].firstnodeborder; //161517;
                imesh+=1; imesh = imesh%MM[0].firstnodeborder; //161517;
                //imeshtestcontinue=false;   // always break out of this while

               /*
                imesh+=17; imesh = imesh%MM[0].node;
                // during first pass include all nodes.  else exclude border nodes 
                if (ip< MM[0].node)  imesh=imesh%MM[0].node;
                else imesh=imesh%MM[0].firstnodeborder;
                */
                // test if inside the box  if it is set continue to false
                if ( MM[0].Lon[imesh]>lonm && MM[0].Lon[imesh]<lonp && 
                     MM[0].Lat[imesh]>latm && MM[0].Lat[imesh]<latp ) 
                     imeshtestcontinue=false;
            }
            host_P[ip].age = 1;  // already launched
            host_P[ip].Release_time = MM[0].time_init + CC->ReleaseIncrement*ip;  // all
            host_P[ip].p_id = ip;
            host_P[ip].i_ele = 55;               
            host_P[ip].WfCore=CC->WfCore;      // Wf=WfCore*cos(time_now*WfFreq - WfShft);
            host_P[ip].WfFreq=CC->WfFreq;      //  2*pi/(Period sec)
            host_P[ip].WfShft=CC->WfShft;
            
            for (int i=0; i<4;i++) host_P[ip].i_ele4[i] = 55;
            if (imesh<MM[0].firstnodeborder) 
                {  
                host_P[ip].state = 3;  // waiting for Release_time 
                host_P[ip].x_present = MM[0].X[imesh] + CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);  
                host_P[ip].y_present = MM[0].Y[imesh] + CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);  
                host_P[ip].z_present = 0.0;    //MM[0].depth[imesh]/2.;
                host_P[ip].x_present = 0.0;  
                host_P[ip].y_present = 0.0;  
                host_P[ip].z_present = 0.0;  //-3.0;    //MM[0].depth[imesh]/2.;
                host_P[ip].XYZstart[0]= MM[0].X[imesh] + CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);
                host_P[ip].XYZstart[1]= MM[0].Y[imesh] + CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);
                host_P[ip].XYZstart[2]= -1.0; //-3.0+ 0.0*MM[0].depth[imesh]; 
            //-1.0;  // -2.0  for a long time now, but...
                //host_P[ip].z_present = MM[0].Y[imesh]/5.;
                }
                else
                {
                    host_P[ip].state = 0;  // white boundary, never move
                    host_P[ip].x_present = MM[0].X[imesh] ;  
                    host_P[ip].y_present = MM[0].Y[imesh] ; 
                    host_P[ip].z_present = 0.0;    //MM[0].depth[imesh]/2.;

                }
        int ippr = 5000;
        if (ip%ippr==0){
            printf("XYDepth[%d,%d]= %f %f %f \n", ip,imesh,MM[0].Lon[imesh],MM[0].Lat[imesh],MM[0].depth[imesh]);
        }        
        }
    } break;
    case CC->Layouts::BorderBay:{
        //CC->shadervs="shaderborder.vs"; 
        //CC->shaderfs="shaderborder.fs";        
        //case  fill bay with a jiggle at nodal points
        printf("PPartInit  MM[0].firstnodeborder = %d   MM[0].nodes=%d \n", MM[0].firstnodeborder,MM[0].node);
        //MM[0].firstnodeborder=99827;
        int ipp = 0;
        int iMM = 0; 
    //for (int iMM=0; iMM<3; iMM++)  {
        int fn=MM[iMM].firstnodeborder;
        for (int ie=0; ie<NELE; ie++)
        {  // loop over all elements looking for ones with nodes in outside list, >firstnodeborder 
        int imesh = -1;
            if (MM[iMM].ele[ie][0]>fn || MM[iMM].ele[ie][1]>fn|| MM[iMM].ele[ie][2]>fn)
            { if (MM[iMM].ele[ie][0]<fn) { imesh = MM[iMM].ele[ie][0];}
              if (MM[iMM].ele[ie][1]<fn) { imesh = MM[iMM].ele[ie][1];}
              if (MM[iMM].ele[ie][2]<fn) { imesh = MM[iMM].ele[ie][2];}
            }
            if (imesh>=0) {
            host_P[ipp].num_P = num_P;
            host_P[ipp].p_id = ipp;
            host_P[ipp].i_ele = ie;
            host_P[ipp].x_present = MM[iMM].X[imesh] ;  
            host_P[ipp].y_present = MM[iMM].Y[imesh] ;  
            host_P[ipp].z_present = 2.0;
            host_P[ipp].state = 0;   //Ever white boundary node
            ipp++;    
            }
        
        }
        //}
            int IPFIRST = ipp;

    // loop over the mesh nodes, several times adding P[ip] at mesh node to the PP list
        int imesh =0;     //  99827  108049;
        for (int ip=IPFIRST; ip<num_P; ip++) {
            host_P[ip].num_P = num_P;
            host_P[ip].p_id = ip;
            host_P[ip].i_ele = 55;
            for (int i=0; i<4;i++) host_P[ip].i_ele4[i] = 55;
                host_P[ip].x_present = MM[0].X[imesh] + 0.0*CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);  
                host_P[ip].y_present = MM[0].Y[imesh] + 0.0*CC->rand_spread*(rand()/(float)RAND_MAX - 0.5);  
                host_P[ip].z_present = -2.0;    //MM[0].depth[imesh]/2.;
                //host_P[ip].z_present = MM[0].Y[imesh]/5.;
                if (imesh<MM[0].firstnodeborder) 
                {  
                    host_P[ip].state = 1;  // moving 
                }
                else
                {
                    //printf("wp=%d ",ip);
                    host_P[ip].state = 0;  // white boundary, never move
                }
                imesh+=1; 
                //imesh=imesh%MM[0].firstnodeborder;
                imesh=imesh%MM[0].node;
                
        }
    } break;
    default: { printf("error particle case");}
}  // switch closure

    cout <<" Particle PPartInit CC->shadervs = "<< CC->shadervs << endl;
    cout <<" Particle PPartInit CC->shaderfs = "<< CC->shaderfs << endl;

}




