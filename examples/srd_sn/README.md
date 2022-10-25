## Hubble Diagram

The structure of input dataset follows the same pattern as
Pantheon dataset for cosmosis. 
This consists of a single systematic covariance matrix and 
the data vector which is the Hubble Diagram (HD) data.

Two types of file structure are allowed at
the moment.

1. COSMOMC Format: The data format can be mostly zeros. The only columns that we actually use values from are:
    ```
    zcmb zhel mb dmb
    ```
    (redshift (z) in CMB frame, redshift in heliocentric frame, magnitude, and magnitude error).
    Each row corresponds to one SN1A in case of unbinned HD or each redshift bin in case of z-binned HD.
    
    The Hubble Diagram data format should contain the following columns:
    ```
    #name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar cov_m_s cov_m_c cov_s_c set ra dec biascor
    ```

2. LSST DESC FORMAT: DESC is building its own Hubble diagram file format. This file format is as follows:
   ```
   # MU        = distance modulus corrected for bias and contamination
   # MUERR     = stat-uncertainty on MU
   # MUERR_SYS = sqrt(COVSYS_DIAG) for 'ALL' sys (diagnostic)
   # ISDATA_REAL: 0   # flag for cosmology fitter to choose blind option.
   #
   VARNAMES: ROW zCMB zHEL MU MUERR MUERR_SYS
   ROW: 1       0.04130 0.04130 36.31576  0.00966   0.00683
   ```

Either of the above formats can be passed as argument for input Hubble Diagram.
 
## Covariance Matrix

The file format has the first line as an integer
indicating the number of supernovae (N) and the subsequent
lines being the elements.
It contains flattened 1-D values of the original square matrix, 
of dimension  = (N X N), where N = number of rows in Hubble Diagram.


## Generating data

`generate_sn_data.py` : creates the sacc file output.

For default execution:
```
python generate_sn_data.py
```

This will generate a folder `sndata` and download the `LSST_DESC_SRD_v1_release` data. 
It will create two sub directories:

1. `Y1_DDF_FOUNDATION`
2. `Y10_DDF_WFD_FOUNDATION`

It will further generate the `sacc` file using the Y1 data.

For user defined input files, 3 arguments are passed.

SYNTAX:
```bash
python generate_sn_data.py <PATH> <Hubble Diagram> <Covariance Matrix>
```

EXAMPLE:
```bash
python generate_sn_data.py ~/example data.txt sys.txt
```

This will generate a `sacc` file with the following namenclature : `<Covariance Matrix>.sacc`

REQUIREMENTS:

Both the files Hubble Diagram and Covariance Matrix should be in the same folder.
The location of which is then passed as argument `<PATH>`

## Likelihood input

`sn_srd.py` : reads in the `sacc` file for generating the SN likelihood object.

User can pass a `sacc` data file as an argument in the following format :
```
python sn_srd.py input_SN_sacc_file.sacc
```
If no argument is provided, the default `sacc` file `srd-y1-converted.sacc` is read. 
