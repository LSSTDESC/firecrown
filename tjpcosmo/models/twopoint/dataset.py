from ...dataset import BaseDataSet
import sacc
import numpy as np

class TwoPointDataSet(BaseDataSet):
    def __init__(self, sacc_data,config) :
        indices=twopoint_process_sacc(sacc_data,config)
        self.data_vector = sacc_data.mean.vector[indices]
        self.covariance = sacc_data.precision.cmatrix[indices,:][:,indices]
        self.precision = np.linalg.inv(self.covariance) #TODO: optimize this through Cholesky

    @classmethod
    def load(cls, data_info, config):
        sacc_data = sacc.SACC.loadFromHDF(data_info['twopoint']['sacc_file'])
        twp=cls(sacc_data,config)
        return twp,config

def twopoint_process_sacc(sacc_data,config):
    tracer_sorting=sacc_data.sortTracers()

    t1_list=np.array([s[0] for s in tracer_sorting])
    t2_list=np.array([s[1] for s in tracer_sorting])
    ell_list=[s[2] for s in tracer_sorting]
    ndx_list=[s[3][0] for s in tracer_sorting]

    tracer_numbers={}
    for itr,tr in enumerate(sacc_data.tracers) :
        tracer_numbers[str(tr.name.decode())]=itr

    for name,d in config['sources'].items() :
        t=sacc_data.tracers[tracer_numbers[name]]
        d['nz']=[t.z,t.Nz]

    indices=[]
    pair_ordering=[]
    for xcor,d in config['statistics'].items() :
        tns=sorted([tracer_numbers[n] for n in d['source_names']])
        id_xcor=np.where((t1_list==tns[0]) & (t2_list==tns[1]))[0]
        if len(id_xcor)==0 :
            raise ValueError("This correlation is not present in the SACC file")
        elif len(id_xcor)!=1 :
            raise ValueError("This SACC file is wrong, the correlation appears more than once")
        else :
            id_xcor=id_xcor[0]
            ells_full=ell_list[id_xcor]
            ndxs_full=ndx_list[id_xcor]
            indices_cut=np.where((ells_full<=d['ell_max']) & (ells_full>=d['ell_min']))[0]
            indices+=ndxs_full[indices_cut].tolist()
            pair_ordering.append({'src1':d['source_names'][0],'src2':d['source_names'][1],'ells':ells_full[indices_cut]})

    config['2pt_ordering']=pair_ordering

    return np.array(indices)
