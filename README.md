# TJPCosmo

TJPCosmo is the nascent LSST/DESC cosmological parameter estimation code.  It is built on the [Core Cosmology Library](https://github.com/LSSTDESC/CCL), [CLASS](http://class-code.net/), and [CosmoSIS](https://bitbucket.org/joezuntz/cosmosis).


## Installation 

TJPCosmo requires python 3.6 or above

Installation is currently a bit painful we will try to improve this in future.
On local machines (where you have root access) you can get a much easier installation using Docker.

## Docker installation

If you install [Docker from here](https://www.docker.com/community-edition) you can get a TJPCosmo-ready docker environment with:

    docker pull joezuntz/tjpcosmo
    git clone https://github.com/LSSTDESC/TJPCosmo
    cd TJPCosmo
    docker run --rm -it -v $PWD:/opt/TJPCosmo joezuntz/tjpcosmo
    cd /opt/TJPCosmo

You will now be in a docker container ready to run TJPCosmo.

## Manual installation

If you don't want to or can't use Docker you can follow these instaructions

### Step 1: SACC

You need to first install [HDF5](https://support.hdfgroup.org/HDF5/) - this is available from that site or in most packagd managers like homebrew, apt, or yum.and then h5py:
    
    pip install h5py

You also need to run these:

    git clone https://github.com/LSSTDESC/sacc/
    cd sacc
    git checkout tjpcosmo_mods
    python setup.py install


### Step 2: CosmoSIS-standalone

Next you can install CosmoSIS.  You need to have newish compilers to make this work - in particular the default Apple Clang with most machines will not work.  You can get newer compilers from [HomeBrew](https://brew.sh/).

Assuming you have installed, for example, gcc version 7, you can install cosmosis like this:

    export CC=gcc
    export FC=gfortran
    export CXX=g++
    pip install cosmosis-standalone


### TJPCosmo

You can now install TJPCosmo:

    git clone https://github.com/LSSTDESC/TJPCosmo
    cd TJPCosmo
    python setup.py install




## A basic test

Now you're ready to run TJPCosmo:

    export PYTHONPATH=$PYTHONPATH:$PWD
    ./bin/tjpcosmo test/3x2pt.yaml 

## LICENSE

The firecrown package is still under development and should be considered work in progress. 
If you make use of any of the ideas or software in this package in your own research, please cite them as 
"(LSST DESC, in preparation)" and provide a link to this repository: https://github.com/LSSTDESC/firecrown. 
If you have comments, questions, or feedback, please [make an issue](https://github.com/LSSTDESC/firecrown/issues).

firecrown calls the CCL library: https://github.com/LSSTDESC/CCL, which makes use of `CLASS`. For free use of the 
`CLASS` library, the `CLASS` developers require that the `CLASS` paper be cited: 
CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933, JCAP 1107 (2011) 034. 
The `CLASS` repository can be found in http://class-code.net. CCL also uses code from the 
[FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We have obtained permission from the FFTLog 
author to include modified versions of his source code.
