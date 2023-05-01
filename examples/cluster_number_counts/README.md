# Cluster Number Counts 

The code here will run a test sample for a Flat LCDM cosmology using CosmoSIS. 

## Running CosmoSIS

The pipeline configuration file `number_counts.ini` and the related `number_counts_values.ini` configure CosmoSIS to use the
`number_counts.py` likelihood, which uses data from true mass and true redshift from the clusters.
Run this using:

    cosmosis number_counts.ini
This will produce the output to the screen, showing the calculated likelihood.
It also creates the directory `output` which is populated with the CosmoSIS datablock contents for the generated sample.
The pipeline configuration file `number_counts_rich.ini` and the related `number_counts_values_rich.ini` configure CosmoSIS to use
`number_counts_rich.py` likelihood, which uses data of richness as mass proxy and true redshift from the clusters.
Run this using:

    cosmosis number_counts_rich.ini

This will produce the output to the screen, showing the calculated likelihood.
It also creates the directory `output_rich` which is populated with the CosmoSIS datablock contents for the generated sample.

