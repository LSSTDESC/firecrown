The structure of input dataset follows the same pattern as
Pantheon dataset for cosmosis. 
This consists of a single systematic covariance matrix and 
the data vector which is the Hubble Diagram data.

Hubble Diagram : The data format can be mostly zeros.
The only columns that we actually use values from are:
  zcmb zhel mb dmb
(redshift in CMB frame, redshift in heliocentric frame, magnitude, and magnitude error).
Each row corresponds to one SN1A.

The Hubble Diagram data format should contain the following columns :
#name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar cov_m_s cov_m_c cov_s_c set ra dec biascor

Covariance Matrix : The file format has the first line as an integer
indicating the number of supernovae (N) and the subsequent
lines being the elements.
It contains flattened 1-D values of the original square matrix, 
of dimension  = (N X N), where N = No. of rows in Hubble Diagram.


`generate_sn_data.py` : creates the sacc file output.

 For default execution,
        `python generate_sn_data.py`

  This will generate a folder `sndata` and download the `LSST_DESC_SRD_v1_release` data. 
  It will create two sub directories :
        a) Y1_DDF_FOUNDATION
   	b) Y10_DDF_WFD_FOUNDATION

    	It will further generate the `sacc` file using the Y1 data.

 For user defined input files,
        3 arguments are passed

  SYNTAX:
	python generate_sn_data.py <PATH> <Hubble Diagram> <Covariance Matrix>

  EXAMPLE:
	`python generate_sn_data.py ~/example data.txt sys.txt`

  REQUIREMENTS:
	Both the files Hubble Diagram and Covariance Matrix should be in the same folder.
	The location of which is then passed as argument <PATH>
