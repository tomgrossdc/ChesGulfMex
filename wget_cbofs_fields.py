# wget_cbofs_fields_2020May.py
#  May 2020   NOS redid their file system
# Files are now in locations like:
# https://opendap.co-ops.nos.noaa.gov/thredds/fileServer/NOAA/CBOFS/MODELS
# /2020/05/21/nos.cbofs.fields.n006.20200520.t18z.nc

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
from pathlib import Path
from subprocess import call
#call(["ls","-1"])

#wwwname="https://opendap.co-ops.nos.noaa.gov/netcdf/cbofs/"
wwwname="https://opendap.co-ops.nos.noaa.gov/thredds/fileServer/NOAA/CBOFS/MODELS/"
filename="nos.cbofs.fields."

# Grab these days in the recent past: (conditionally if they haven't been grabbed yet, of course)
for ipastdays in [12,11,10,9,8,7,6,5,4,3,2,1]:
	ti = time.gmtime(time.time()-ipastdays*24*60*60)
	dateget="%d%02d%02d."%(ti.tm_year ,ti.tm_mon,ti.tm_mday)
	yearn="%4d%02d/"%(ti.tm_year,ti.tm_mon)

	#Local directory name
	dirname="/media/tom/MyBookAllLinux/NOSnetcdf/"+yearn

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
			
	
	for sixget in ["t00z.", "t06z.", "t12z.", "t18z.", ] :
		for nh in ["n001.","n002.","n003.","n004.","n005.","n006."] :
			#print(type(filen))
			filen=filename+nh+dateget+sixget+"nc"
			filenew=filename+dateget+sixget+nh+"nc"
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





