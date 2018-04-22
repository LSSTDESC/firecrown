FROM python:3.6
MAINTAINER joezuntz@googlemail.com
#Joe's note to himself.  Compile this with: docker build -t joezuntz/cosmosis-base .
#then docker push joezuntz/cosmosis-base

# Basic compilers and tools dependencies
RUN apt-get update -y && apt-get install -y gcc g++ gfortran wget make  \
    pkg-config curl \
    && apt-get clean all

# Manual installation of mpich seems to be required to work on NERSC
RUN mkdir /opt/mpich && cd /opt/mpich \
    && wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
    && tar xvzf mpich-3.2.tar.gz &&  cd mpich-3.2 && ./configure --disable-wrapper-rpath && make -j4 \
    && make install && rm -rf /opt/mpich

RUN mkdir /opt/gsl && cd /opt/gsl \
    && wget http://mirror.koddos.net/gnu/gsl/gsl-2.4.tar.gz \
    && tar xvzf gsl-2.4.tar.gz \
    && cd gsl-2.4 \
    && ./configure && make && make install \
    && rm -rf /opt/gsl

# other dependencies, some for matplotlib
RUN apt-get update  && \
	apt-get install -y  libcfitsio-dev  libfftw3-dev   git  \
    && apt-get clean all

RUN mkdir /opt/hdf && cd /opt/hdf \
    && wget -O hdf5-1.10.2.tar.gz "https://www.hdfgroup.org/package/source-gzip-2/?wpdmdl=11810&refresh=5ad9be5a30b831524219482" \
    && tar xvf hdf5-1.10.2.tar.gz \
    && cd hdf5-1.10.2 \
    && ./configure --enable-parallel --prefix=/usr/local CC=mpicc \
    && make \
    && make install \
    && rm -rf /opt/hdf



# Also need a manual install of mpi4py so that it uses the right libraries - pip seems not to work
RUN mkdir /opt/mpi4py && cd /opt/mpi4py \
&& wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz \
&& tar -zxvf mpi4py-3.0.0.tar.gz && cd mpi4py-3.0.0 && python setup.py install \
&& rm -rf /opt/mpi4py



# The environment variables needed by the CosmoSIS build and runtime.

ENV PYTHONPATH ${COSMOSIS_SRC_DIR}:${PYTHONPATH}
ENV PATH ${COSMOSIS_SRC_DIR}/bin:${PATH}
ENV HDF5_MPI ON

# Ceci
RUN apt-get install -y libopenblas-dev && apt-get clean all

RUN pip install --upgrade pip

RUN pip install Cython==0.28.2 \
                nose==1.3.7 \
                numpy==1.14.2 \
                pyparsing==2.2.0 \
                pyyaml==3.12 \
                scikit-learn==0.19.1 \
                setuptools==38.5.2

RUN pip install astropy==3.0.1 \
                emcee==2.2.1 \
                scipy==1.0.1 \
                scikit-learn==0.19 \
                fitsio==0.9.11 \
                healpy==1.11.0  \
                ceci


RUN pip install --no-binary=h5py h5py

RUN cd /opt/ && git clone https://github.com/LSSTDESC/CCL \
&& cd CCL && python class_install.py \
&& ./configure && make && make install \
&& python setup.py install --prefix=/usr/local

RUN CC=gcc CXX=g++ FC=gfortran pip install cosmosis-standalone

RUN pip install pyccl

RUN cd /opt && git clone https://github.com/LSSTDESC/sacc/ && cd sacc && python setup.py install

# Run a bash login shell if no other command is specified.
CMD ["/bin/bash", "-l"]
