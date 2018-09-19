from ...dataset import BaseDataSet
import sacc
import numpy as np


class TwoPointDataSet(BaseDataSet):
    def __init__(self, sacc_data, indices):
        self.data_vector = sacc_data.mean.vector[indices]
        self.covariance = sacc_data.precision.cmatrix[indices, :][:, indices]
        # TODO: optimize this through Cholesky
        self.precision = np.linalg.inv(self.covariance)
        self.nsims = sacc_data.meta.get("nsims", 0)

    @classmethod
    def load(cls, filename, config):
        sacc_data = sacc.SACC.loadFromHDF(filename)
        indices, metadata = twopoint_process_sacc(sacc_data, config)
        twp = cls(sacc_data, indices)
        return twp, metadata


def twopoint_process_sacc(sacc_data, config):
    tracer_sorting = sacc_data.sortTracers()

    metadata = {}
    metadata['sources'] = {}

    t1_list = np.array([s[0] for s in tracer_sorting])
    t2_list = np.array([s[1] for s in tracer_sorting])
    typ_list = np.array([s[2].decode() for s in tracer_sorting])
    xs_list = [s[3] for s in tracer_sorting]
    ndx_list = [s[4] for s in tracer_sorting]

    tracer_numbers = {}
    for itr, tr in enumerate(sacc_data.tracers):
        tracer_numbers[str(eval(tr.name).decode())] = itr

    for name, d in config['sources'].items():
        t = sacc_data.tracers[tracer_numbers[name]]
        metadata['sources'][name] = {'z': t.z, 'nz': t.Nz}

    indices = []
    pair_ordering = []
    # Dictionary between SACC 2-point types and TJPCosmo 2-point types
    dict_types = {
        'ClGG': 'FF',
        'ClGE': 'FF',
        'ClEE': 'FF',
        'XiGG': '+R',
        'XiGE': '+R',
        'XiP': '+R',
        'XiM': '-R'}
    for xcor, d in config['statistics'].items():
        print("xcor = ", xcor)
        tns = sorted([tracer_numbers[n] for n in d['source_names']])
        typ = dict_types[d['type']]
        id_xcor = np.where(
            (t1_list == tns[0]) & (t2_list == tns[1]) & (typ_list == typ))[0]
        if len(id_xcor) == 0:
            raise ValueError(
                f"The correlation {xcor} is not present in the SACC file")
        elif len(id_xcor) != 1:
            raise ValueError(
                f"This SACC file is wrong, the correlation "
                f"{xcor} appears more than once")
        else:
            id_xcor = id_xcor[0]
            xs_full = xs_list[id_xcor]
            ndxs_full = ndx_list[id_xcor]
            indices_cut = np.where(
                (xs_full <= d['x_max']) & (xs_full >= d['x_min']))[0]
            indices += ndxs_full[indices_cut].tolist()
            pair_ordering.append({
                'name': xcor,
                'src1': d['source_names'][0],
                'src2': d['source_names'][1],
                'type': d['type'],
                'xs': xs_full[indices_cut],
                })
    metadata['ordering'] = pair_ordering
    return np.array(indices), metadata
