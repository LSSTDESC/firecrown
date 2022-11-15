# Documentation for the SACC format needed for number_counts_stats.py

The Number Counts statistics object needs to be initialized with a SACC file in the right format. An example of how to create a file in such format can be found in `firecrown/examples/number_counts/Generating Cluster Data.ipynb`. Here we discuss how the file must be created.

The user must provide a SACC file that only has one tracer, which shall be called `cluster_counts_true_mass`. So far, firecrown only has the implementation for real mass and real redshift. More options will be provided in the future. The tracer must be added as being the type `misc` and should be created with a directory to be added under `metadata`.

The metadata dictionary must be created with:
*`Mproxy_type: 'true_mass'`. A `str` that represents the type of proxy. So far there is only the `true_mass` option.
*`Mproxy_edges : m_edges`. A `list` of `floats` with the proxy bins.
*`z_type : 'true_redshift'`. A string that represents the type of redshift. So far there is only the `true_redshift` option.
*`z_edges : z_edges`. A `list` of `floats` with the redshift bins.
*`sky_area: 439.78986`. A `float` with the sky area.  

With the metadata, the `add_tracer` function must be called as
```
s_count.add_tracer('misc', 'cluster_counts_true_mass', metadata=metadata).
```

 Having the tracer, the user must call the `add_data` function for all the data points using the same tracer. The data should be the number of clusters in each bin of redsfhit and proxy. The order of the data is really important: the user must add the data for each redshift bin and then for the proxy bins. Example, for 3 bins of mass: data1 = (z_bin=0, proxy_bin = 0), data2 = (z_bin=0, proxy_bin = 1), data3 =(z_bin=0, proxy_bin = 2), data4 = (z_bin=1, proxy_bin = 0), etc.
The function to call the data must be such as
```
cluster_count = sacc.standard_types.cluster_mass_count_wl
for i in range(z_bins):
    for i in range(proxy_bins):
        tracer = 'cluster_counts_true_mass'
        value = data_lz[i][j]
        s_count.add_data_point(cluster_count, (tracer,), value, err=1.)
```
In the above, the `cluster_count` variable represents the type of data point. The SACC library already has some `standard_types` and thus the data should be added with the `cluster_mass_count_wl` option. The `value` should be the number counts data in each bin of redsfhit and proxy.

 Lastly, the user must add the covariance as a `numpy.ndarray` and call the following functions to save the file:
 ```
 s_count.add_covariance(covariance)
 s_count.to_canonical_order()
 s_count.save_fits("clusters.sacc", overwrite=True)
 ```
Other than the Number counts data, there should not be other tracers or data in the file.

# Documentation for number_counts_stats.py

This module reads the necessary data from a SACC file to compute the theoretical prediction of cluster number counts inside bins of redshift and a mass proxy. The `NumberCountStat` class must be initialized with:

*`sacc_tracer : str`. The user must provide the tracer's name. Following the documentation above it should be a `cluster_counts_true_mass` string.
*`sacc_data_type : str`. The type of SACC data. So far, following the documentation above, this entry should be a string  `cluster_mass_count_wl`.
*`systematics: Optional[List[Systematic]]`. This entry is optional. This option represents the implementation of a shift or a bias in the redshift or the proxy. No method utilizes this option so far in this class.

## Read Function
The read function takes a SACC file and extracts the necessary data to perform a cluster number counts analysis. With some pre-defined functions from the SACC library, we extract:

*The `metadata` from the `sacc_tracer` given by the user. It should contain the redshift and proxy bins and types, and also the sky area.
So far, there are only implementations for `true_redshift` and `true_mass` for the types. Check the above documentation for further information on the `metadata`.
*The `nz`, that is, the number of clusters in each bin, which is stored in the `data.value` in the SACC file.

After the information is extracted, the `data.value` is stored as the `self.data_vector` variable and the rest of the information is stored under an internal class `NumberCountStatsArgs` that will be passed to `self.tracer_args`.

Other than the Number counts data, there should not be other tracers or data in the file.

##Compute function

This function utilizes the data from the `read` method to compute a theoretical prediction of the cluster number counts in each bin. This implementation takes a `pyccl.Cosmology` and utilizes the
`pyccl.halos.MassFuncBocquet16` as the dark matter halo mass function to compute the theoretical prediction.

The user shall provide the chosen cosmology and the necessary data in the SACC file so that this method can be used. So far, there is only the Tinker mass function implemented.

This function returns the theoretical prediction as a Numpy vector in `theory_vector` and the data points as a Numpy vector in `data_vector`.
