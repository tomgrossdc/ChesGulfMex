# wget_ngofs_fields.py
#  July 2020   Reading Gulf of Mexico Files
# They are about 20 times bigger than CBOFS !  Yipes

# Files are now in locations like:

# https://opendap.co-ops.nos.noaa.gov/thredds/fileServer/NOAA/NGOFS/MODELS/2020/07/02/nos.ngofs.fields.n006.20200702.t21z.nc


# https://opendap.co-ops.nos.noaa.gov/thredds/fileServer/NOAA/GOMOFS/MODELS/2020/06/27/nos.gomofs.fields.n006.20200627.t18z.nc

# To grab real netcdf files, even though they are only six hourly
# Build a script to do this:
# wget https://opendap.co-ops.nos.noaa.gov/netcdf/cbofs/201905/nos.cbofs.fields.n003.20190510.t18z.nc
# only can get t00z, t06z, t12z, t18z
# 48 hour forecasts are also available:
# wget  https://opendap.co-ops.nos.noaa.gov/netcdf/cbofs/201905/nos.cbofs.fields.f048.20190511.t12z.nc

# the Web site to visit is https://opendap.co-ops.nos.noaa.gov/index.php?dir=/netcdf/cbofs/201807/
# subprocess http://www.pythonforbeginners.com/os/subprocess-for-system-administrators

# mkdir 201807
# cd 201807

import subprocess
import shutil, os
import time
import sys
from pathlib import Path
from subprocess import call
#call(["ls","-1"])

print ('python3 wget_ofs_fields.py nwgofs')
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

gofs=str(sys.argv[1]).lower()
#gofs="nwgofs"
GOFS=str(sys.argv[1]).upper()

#wwwname="https://opendap.co-ops.nos.noaa.gov/netcdf/cbofs/"
wwwname="https://opendap.co-ops.nos.noaa.gov/thredds/fileServer/NOAA/"+GOFS+"/MODELS/"
filename="nos."+gofs+".fields."

# Grab these days in the recent past: (conditionally if they haven't been grabbed yet, of course)
for ipastdays in [28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]:
#for ipastdays in [4,3]:
	ti = time.gmtime(time.time()-ipastdays*24*60*60)
	dateget="%d%02d%02d."%(ti.tm_year ,ti.tm_mon,ti.tm_mday)
	yearn="%4d%02d/"%(ti.tm_year,ti.tm_mon)

	#Local directory name
	dirname="/media/tom/A8TLinux/NOSnetcdf/"+GOFS+"/"+yearn

	print ("\n\n\n\nTime yearn : ", yearn)
	print ("Time dateget : ", dateget)
	print
	# old NOS dirname 202005

	# new threads dirname 2020/05/21
	yearnd="%4d/%02d/%02d/"%(ti.tm_year,ti.tm_mon,ti.tm_mday)

	#os.makedirs(dirname,exist_ok=True)
	try:
		call(["mkdir",dirname])
	except :
		print("Local Monthly directory already exists")
			
	



#	for sixget in ["t03z.", "t09z.", "t15z.", "t21z.", ] :
#		for nh in ["n000.","n001.","n002.","n003.","n004.","n005.","n006."] :
# t03z.n003 is 0:00  six hours off. Damn names of files are in Eastern Time! not UTC
# likely to change in future daylight savings 
# Well, the particle tracker reads the real datetime as UTC, so not real problem.
	for thour in [3,9,15,21]:
		sixget = 't'+'{:02d}'.format(thour)+'z.'
		for nhour in [1,2,3,4,5,6] :
			nh = 'n'+'{:03d}'.format(nhour)+'.'
			#     nos.ngofs.fields.n006.20200702.t21z.nc
			filen=filename+        nh+  dateget+ sixget+"nc"

			n24hour = thour +nhour -4 
			if n24hour==24: 
				n24hour=0

			n24h='n'+'{:03d}'.format(n24hour)
			#print(type(filen))

			filenew=filename+dateget+n24h+".nc"
			#if nhour==6:
			#	filenew=filename+dateget+n24h+"X.nc"
			#if nhour==0:
			#	filenew=filename+dateget+n24h+"Y.nc"

			testfile=Path(dirname+filenew)
			if testfile.exists() == False:
				#file does not exist, make a new one
				#print("filen="+filen)
				#print ("wwwname="+wwwname)

				calllist=["wget",wwwname+yearnd+filen]
				print (calllist)
				call(calllist)

				calllistmv=["mv",filen,dirname+filenew]
				print ( calllistmv)
				call(calllistmv)
			else:
				print("EXISTS ",dirname+filenew," EXISTS")

ti = time.gmtime()
print ("Time ending  is : "+time.ctime())




