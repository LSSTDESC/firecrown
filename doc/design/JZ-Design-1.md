TJPCosmo Architecture Proposal
==============================

The diagram JZ-Design-1.png shows a proposed design for the TJPCosmo software.  Below I describe the conventions of the diagram and detail the roles of the different components.

Comments please!
Joe Zuntz


Diagram Conventions
===================

Top part = setup phase, at the start of execution.
Bottom part = execution phase

Solid border = python classes (or perhaps functions in a few cases)
Dotted border = input data files
Dashed border = collection of models in various classes/files

No fill = new code
Yellow fill = already implemented in cosmosis
Green = other existing libraries (just CCL for now)

The CosmoSIS components will be pip installable from the (nearly ready) standalone version of the code (doesn't include the standard library).  If there are any deficiencies in it I can work on them.



Component Reference: New Classes
================================


Main
----

The main object would be the primary entry point to the command line user interface, but it should be separate from it and scriptable as I've found that to be useful for many situations.

Tasks:
 - runs the Config Reader and gets back Config Data
 - instantiates the Theory Calculator, Likelihood Calculator, Sampler(s), Parallel Pool (if in parallel), and Outputter using bits of the Config Data
 - calls the sampler(s) when ready


Theory Calculator
-----------------

The theory calculator is chosen from the library of python models which we should supply and enable people to write variants and extensions of.  Writing these calculators is the bulk of the work in TJPCosmo!

The various mockups currently in TJPCosmo/sandbox show a range of design ideas for this component.

Tasks:
- configured first by choosing a particular model from a library
- then configured with any additional settings from input files, including possibly a DataSet object so that it knows what e.g. bins, angles, redshifts to calculate.
- takes Parameters object as input.
- returns Theory Results object as output.
- calls CCL to do its cosmology calculations.
- calls probe-specific models in other packages (or included?) to do its probe-specific systematic calculations.


Theory Results
--------------

The theory results objects encapsulates the results of theory calculations for a specific model. I would suggest the following design:

- A parent class with some basic behaviour (see below).
- Each theory calculator defines its own subclass with its results in.
- Slots are defined for each calculated theory result in a given model  This should *not* just consist of the theory vector elements, but also the steps leading up to that (e.g. matter power spectrum) because these are critical for code testing and making plots later.

Base class tasks:
- management of data slots structure, including setters and getters.
- saving all data slots to disk (could use cosmosis system for this).
- generic interpolation and windowing methods for subclasses to use.

Subclass tasks:
- define data slots needed for this model
- extract required theory vector elements in correct order (requires DataSet metadata)

Examples:
- SupernovaResults: has slots for tables of z, mu(z), and other distances calculated with mu.  When passed a DataSet that contains specific observed redshifts it would interpolate mu to the correct redshifts
- LSSResults: has slots for distances, P(k), and C_ell with metadata showing which bin pair, and sample it refers to.  When given a DataSet that includes a window function it integrates over that window function to get C_bin.

I haven't thought about how this would fit in with a more complex hierarchical model. Thoughts especially welcome.


Likelihood Calculator
---------------------

The likelihood calculator combines a Theory Results object with a DataSet object to obtain a likelihood.  In this simplest case this would just be Gaussian, but given the interesting current work in the community on likelihoods this must be flexible.

Tasks:
- call the Theory Results object with each DataSet to obtain a theory prediction.
- generate a likelihood from the theory prediction and the DataSet.


Data files
----------

We should have a well-defined set of data input files designed by the working groups that they commit to generating.  The data files need to include *all* the information that is needed to compute a likelihood given a theoretical model, very explicitly.  There will be no assumptions - we should be able to give this file to a completely separate team with no information about us and they should be able to use them in their own pipelines.

At this stage I would recommend that the data files also include an estimated covariance, but we may want to consider different designs for this.

The format of these files is less important than their having a well-defined schema.

An example is the SACC format being developed for LSS.


DataSet
-------

The DataSet objects encapsulate all the information about measurements made on data, including the metadata needed to generate a theory prediction, like number density data for bins in WL/LSS.  It is assumed at this stage to include a covariance model, but we should talk about that some more.

A given data file as described above corresponds to a specific DataSet subclass.

There should be a method that can be used to generate an internally simulated DataSet from a Theory Results object, for simulation and testing purposes.

Tasks
- load a DataSet from file(s)
- save a DataSet to file(s)
- mask or cut out pieces of data and keep the covariance up to date with this
- generate the metadata required to assemble the various data points into a specific order
- generate a data vector in this specified order




Component Reference: CosmoSIS Classes
=====================================

These classes already exist in CosmoSIS and I would recommend using them as-is or with small modifications to the cosmosis core.

I'm shortly going to release a pip-installable CosmoSIS that would make these all very easy to acquire.

Config Reader
-------------

The configuration reader exists in cosmosis (cosmosis/runtime/config.py) already and reads configuration files in an extended variant of the "ini" format, inheriting from python's standard library configparser.ConfigParser class.

Tasks:
- opens and reads file into a dict-like object (with two keys, section and name)
- interpolates variables according to ini file rules
- replaces environment variables
- if finds a special token includes other files


Configuration Files
-------------------

The current CosmoSIS config file format uses 3 configiuration files, one of them optional:

params: describes the sampler, output, and pipeline.
values: describes the input parameters and their allowed ranges and starting points, including fixed values.
priors: (optional) describes any additional priors on top of the implicit flat prior from the values file.

We could modify this design if useful for DESC.  I've been mulling combining the "values" and "priors" files into one, so that every parameter is created with a single prior.

We could also consider splitting the "params" file into "pipeline" and "sampler/output".



Configuration Data
------------------

The current CosmoSIS configuration data approach is a python DataBlock object (cosmosis/datablock/block.py), which acts like a python dictionary with two keys (section and name) rather than just one.

It also handles typing, and convert inputs into integer, float, or bool form from string when asked for an option of a specific type.


Sampler
-------

The CosmoSIS samplers are all subclasses of base classes Sampler and ParallelSampler (cosmosis/samplers/samplers.py).  

In this context "sampler" does not just refer to MCMC samplers.  Anything that generates one or more sets of parameters given some priors is a "sampler", including a Fisher matrix calculation, grid evaluation, and a test sampler that just runs on one value.

Most of the CosmoSIS samplers would work as-is with the design above.  One possible excption is the Fisher matrix sampler, which might have to be modified to use the theory predictions in a different form.

Tasks:
- choose sets of parameters to run via whatever algorithm
- call the likelihood function
- send output rows to the Output object
- handle using parallel pools
- decide when they have converged


Outputter
---------

The CosmoSIS outputters are all subclasses of a base class Output (cosmosis/output/output_base.py).  The most common one is the text output but there is also a FITS output variant.  It is straightforward to create new ones (e.g. HDF5).

Tasks:
- save metadata information
- save comments
- keep track of expected output (e.g. column names and types)
- lock files to avoid multiple writing errors

Parallel Pool
-------------

Most of the samplers that we use can do parallelism via a "pool" of workers.

The CosmoSIS implementation of this can use either the in-built multiprocessing.Pool or a distributed MPIPool (cosmosis/runtime/mpi_pool.py).


I would recommend that we (well, I ) should update the cosmosis parallel pool implementation to use the Schwimbad python package, which combines various types of parallel pool into the same API.

Tasks:
 - provide a "map" function that takes a function f and list [a,b,c,...] of tasks and return [f(a), f(b), f(c), ...]. After evaluating them in parallel.
 - provide an "is_master" function to check if a process should be the root
 - provide a "wait" function for worker processes to await tasks


Likelihood Function
-------------------

The likelihood function will be implemented as a CosmoSIS "Pipeline" object (cosmosis/runtime/pipeline.py) but with a fixed set of input modules generated programmatically, one for the model and one for the likelihood.

This will be a small wrapper function.


Chains
------

The cosmosis text chain format tries to be as explicit as possible and saves lots of metadata and auxiliary data along with the output.

Although I described it as a chain here the exact content depends on the sampler choice.  e.g. we might do a Fisher matrix in which case it would contain that.