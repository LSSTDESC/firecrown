generate_sn_data.py : creates the sacc file output.

For default execution,
python generate_sn_data.py


For user defined input files,
3 arguments are passed

SYNTAX:
python generate_sn_data.py <PATH> <Hubble Diagram> <Covariance Matrix>

EXAMPLE:
python generate_sn_data.py ~/example data.txt sys.txt

REQUIREMENTS:
Both the files Hubble Diagram and Covariance Matrix should be in the same folder.
The location of which is then passed as argument <PATH>