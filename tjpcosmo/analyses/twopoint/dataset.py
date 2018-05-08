from ...dataset import BaseDataSet
import sacc
import numpy as np
from tjpcosmo.main import parser
import argparse
import re

def readpar(path,par):
    with open(path,'r') as fn:
        for line in fn:
            if '=' in line and line[0] != '#':
                line = re.sub('#.+', '', line)
                name, value = line.split('=')
                name = name.strip()
                if name == par:
                    return value.strip()
        raise ValueError('parameter not found')

class TwoPointDataSet(BaseDataSet):
    def __init__(self, sacc_data,config) :
        indices=twopoint_process_sacc(sacc_data,config)
        self.data_vector = sacc_data.mean.vector[indices]
        self.covariance = sacc_data.precision.cmatrix[indices,:][:,indices]
        self.precision = np.linalg.inv(self.covariance) #TODO: optimize this through Cholesky
        print('likelihood choice: ',readpar(parser.parse_args().inifile,'likelihood'))
        if readpar(parser.parse_args().inifile,'likelihood') =='tdist' :
            self.nsims = sacc_data.meta["nsims"]

    @classmethod
    def load(cls, data_info, config):
        sacc_data = sacc.SACC.loadFromHDF(data_info['twopoint']['sacc_file'])
        twp=cls(sacc_data,config)
        return twp,config

def twopoint_process_sacc(sacc_data,config):
    tracer_sorting=sacc_data.sortTracers()

    t1_list=np.array([s[0] for s in tracer_sorting])
    t2_list=np.array([s[1] for s in tracer_sorting])
    typ_list=np.array([s[2].decode() for s in tracer_sorting]) 
    xs_list=[s[3] for s in tracer_sorting]
    ndx_list=[s[4] for s in tracer_sorting]

    tracer_numbers={}
    for itr,tr in enumerate(sacc_data.tracers) :
        tracer_numbers[str(tr.name.decode())]=itr

    for name,d in config['sources'].items() :
        t=sacc_data.tracers[tracer_numbers[name]]
        d['nz']=[t.z,t.Nz]

    indices=[]
    pair_ordering=[]
    dict_types={'ClGG':'FF','ClGE':'FF','ClEE':'FF','XiGG':'+R','XiGE':'+R','XiP':'+R','XiM':'-R'} #Dictionary between SACC 2-point types and TJPCosmo 2-point types
    for xcor,d in config['statistics'].items() :
        tns=sorted([tracer_numbers[n] for n in d['source_names']])
        typ=dict_types[d['corr_type']]
        id_xcor=np.where((t1_list==tns[0]) & (t2_list==tns[1]) & (typ_list==typ))[0]
        if len(id_xcor)==0 :
            raise ValueError("This correlation is not present in the SACC file")
        elif len(id_xcor)!=1 :
            raise ValueError("This SACC file is wrong, the correlation appears more than once")
        else :
            id_xcor=id_xcor[0]
            xs_full=xs_list[id_xcor]
            ndxs_full=ndx_list[id_xcor]
            indices_cut=np.where((xs_full<=d['x_max']) & (xs_full>=d['x_min']))[0]
            indices+=ndxs_full[indices_cut].tolist()
            pair_ordering.append({'src1':d['source_names'][0],'src2':d['source_names'][1],'type':d['corr_type'],'xs':xs_full[indices_cut]})

    config['2pt_ordering']=pair_ordering

    return np.array(indices)
