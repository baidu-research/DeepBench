#!/bin/sh
# set -ex

FindLibrary() 
{
  LOCALMKL=`find $1 -name libmklml_intel.so`   # name of MKL lib
}

GetVersionName()
{
  VERSION_LINE=0
  if [ $1 ]; then
    VERSION_LINE=`grep __INTEL_MKL_BUILD_DATE $1/include/mkl_version.h 2>/dev/null | sed -e 's/.* //'`
  fi
  if [ -z $VERSION_LINE ]; then
    VERSION_LINE=0
  fi
  echo $VERSION_LINE  # Return Version Line
}

# MKL
DST=`dirname $0`
OMP=0 
VERSION_MATCH=20160906
ARCHIVE_BASENAME=mklml_lnx_2017.0.1.20161005.tgz
MKL_CONTENT_DIR=`echo $ARCHIVE_BASENAME | rev | cut -d "." -f 2- | rev`
GITHUB_RELEASE_TAG=self_containted_MKLGOLD_u1
MKLURL="https://github.com/intel/caffe/releases/download/$GITHUB_RELEASE_TAG/$ARCHIVE_BASENAME"
# there are diffrent MKL lib to be used for GCC and for ICC
reg='^[0-9]+$'
VERSION_LINE=`GetVersionName $MKLROOT`
# Check if MKLROOT is set if positive then set one will be used..
if [ -z $MKLROOT ] || [ $VERSION_LINE -lt $VERSION_MATCH ]; then
  # ..if MKLROOT is not set then check if we have MKL downloaded in proper version
  VERSION_LINE=`GetVersionName $DST/$MKL_CONTENT_DIR`
  if [ $VERSION_LINE -lt $VERSION_MATCH ] ; then
    #...If it is not then downloaded and unpacked
    wget --no-check-certificate -P $DST $MKLURL -O $DST/$ARCHIVE_BASENAME
    tar -xzf $DST/$ARCHIVE_BASENAME -C $DST
  fi
  FindLibrary $DST
  MKLROOT=$PWD/`echo $LOCALMKL | sed -e 's/lib.*$//'`
fi
# Check what MKL lib we have in MKLROOT
if [ -z `find $MKLROOT -name libmkl_rt.so -print -quit` ]; then
  # mkl_rt has not been found; we are dealing with MKLML

  if [ -z $LOCALMKL ] ; then
    # LOCALMKL is not set, when MKLROOT was set manually and it points to MKLML in correct version
    FindLibrary $MKLROOT
  fi

  LIBRARIES=`basename $LOCALMKL | sed -e 's/^.*lib//' | sed -e 's/\.so.*$//'`
  OMP=1
else
  LIBRARIES="mkl_rt"
fi 

echo $MKLROOT $LIBRARIES $OMP
