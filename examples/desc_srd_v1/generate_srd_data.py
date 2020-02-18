import sacc
import firecrown
from firecrown.ccl.two_point import build_sacc_data

# read the config
config, data = firecrown.parse("srd_v1_gen.yaml")

# run the config
cosmo = firecrown.get_ccl_cosmology(config['parameters'])
_, stats = firecrown.compute_loglike(cosmo=cosmo, data=data)
print (data['two_point']['data']['statistics']['cl_src0_src0'])


# build sacc data from outputs
_, pred_sacc = build_sacc_data(
    data['two_point']['data'],
    stats['two_point']
)

# add covariance
srd_sacc = sacc.Sacc.load_fits("srd_data/srd_v1_sacc_data.fits")
pred_sacc.add_covariance(srd_sacc.covariance)

# write to disk
pred_sacc.save_fits('srd_v1_gen_sacc_data.fits', overwrite=True)
