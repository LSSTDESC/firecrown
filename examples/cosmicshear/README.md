# Cosmic Shear Example

To run the example, first generate the data

    python generate_cosmicshear_data.py

This will create the file `cosmicshear.fits`.

Then run CosmoSIS using:

    cosmosis cosmicshear.ini

This uses the `test` sampler, and will write the likelihood output to the screen.
