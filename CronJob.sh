#!/bin/sh
# CronJob.sh
export WGETWORKDIR=/media/tom/A8TLinux/NOSnetcdf/CRONTAB
echo " CRONJob.sh with wget_regular_test.py WGETWORKDIR = ", $WGETWORKDIR
cd $WGETWORKDIR
python3 wget_cbofs_fields.py >> $WGETWORKDIR/junk.log
python3 wget_ofs_fields.py nwgofs>> $WGETWORKDIR/junk.log
python3 wget_ofs_fields.py negofs>> $WGETWORKDIR/junk.log
python3 wget_ofs_fields.py ngofs>> $WGETWORKDIR/junk.log
date >> $WGETWORKDIR/junk.log
echo "THE END"

