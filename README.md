# TJPCosmo

TJPCosmo is the nascent LSST/DESC cosmological parameter estimation code.  It is built on the [Core Cosmology Library](https://github.com/LSSTDESC/CCL), [CLASS](http://class-code.net/), and [CosmoSIS](https://bitbucket.org/joezuntz/cosmosis).


## Installation 

TJPCosmo requires python 3.

TJPCosmo requires various dependencies, which will be installed when you run:

    python setup.py install

You also need SACC - see below

## SACC

The SACC library is currently a little more difficult to install.

You need to first install [HDF5](https://support.hdfgroup.org/HDF5/), and then h5py:
    
    pip install h5py

You also need to run these:

    git clone https://github.com/LSSTDESC/sacc/
    cd sacc
    python setup.py install


## A basic test

Now you're ready to run TJPCosmo
    git clone https://github.com/LSSTDESC/TJPCosmo
    export PYTHONPATH=$PYTHONPATH:PWD
    ./bin/tjpcosmo test/3x2pt.yaml 



## Docker

If you install Docker you can get a TJPCosmo-ready docker environment with:

    docker pull joezuntz/tjpcosmo
    cd /path/to/TJPCosmo
    docker run --rm -it -v $PWD:/opt/TJPCosmo joezuntz/tjpcosmo

You will now be in a docker container and can cd to /opt/TJPCosmo to use TJPCosmo.
