# TJPCosmo



## Docker

If you install Docker you can get a TJPCosmo-ready docker environment with:

    docker pull joezuntz/tjpcosmo
    cd /path/to/TJPCosmo
    docker run --rm -it -v $PWD:/opt/TJPCosmo joezuntz/tjpcosmo

You will now be in a docker container and can cd to /opt/TJPCosmo to use TJPCosmo.


Then you can test with:


    bin/tjpcosmo test/params.ini
