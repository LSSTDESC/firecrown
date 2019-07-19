from ..sources import ClusterSource
import os
bin_data = {
    'z_min':0.1,
    'z_max':1.0,
    'proxy_min':20,
    'proxy_max':40,
    'area_eff':10,
}

f = open('dummy_cl_bin_data.csv', 'w')
print(','.join(bin_data.keys())+'\n'+
        ','.join([str(i) for i in bin_data.values()]),
        file=f)
f.close()
cl = ClusterSource('dummy_cl_bin_data.csv')
os.system('rm dummy_cl_bin_data.csv')
print(cl)

def test_cl_init():
    cl_bin_data = {
        'z_min':cl._z_min,
        'z_max':cl._z_max,
        'proxy_min':cl._proxy_min,
        'proxy_max':cl._proxy_max,
        'area_eff':cl._a_eff,
    }
    assert cl_bin_data == bin_data
