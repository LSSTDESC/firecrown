class Model:
    def __init__(self, cosmology, systematics):
        self.cosmology = cosmology
        self.systematics = systematics

    def parameters(self):
        params = self.cosmology.parameters()
        for sys in self.systematics.parameters():
            params += sys.parameters()
        return parameters

    def input_systematics(self):
        return [sys for sys in self.systematics if isinstance(sys,InputSystematic)]

    def output_systematics(self):
        return [sys for sys in self.systematics if isinstance(sys,OutputSystematic)]

    def calculation_systematics(self):
        return [sys for sys in self.systematics if isinstance(sys,CalculationSystematic)]


class Systematic:
    def run(self, model):
        pass

class InputSystematic:
    pass

class OutputSystematic:
    pass

class CalculationSystematic:
    pass

class TheoryVectorCalculator:
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, parameters):
        """"

        inputs: dict (or maybe some more structured object??) of SourceSamples
        parameters: dict of scalar parameter values (maybe comine this into inputs??)
        # yes, a
        """
        for sys in self.model.input_systematics():
            inputs = sys(inputs)
        
        outputs = self.compute(inputs, parameters)

        for sys in model.output_systematics():
            sys(outputs, parameters)
            

class SourceSample:
    def __init__(self, z, n_of_z): # ...
        pass

class ShearShearCellVector:
    def __init__(self, n1, n2, source1_name, source2_name):
        self.n1 = n1
        self.n2 = n2
        self.source1_name = source1_name
        self.source2_name = source2_name
        self.values = {}

    def add(self, b1, b2, theta, value):
        self.values((b1,b2)) = theta, value

    def __contains__(self, b1_b2):
        return b1_b2 in self.values

    def __getitem__(self, b1_b2):
        return self.values[b1_b2]

class ShearShearCalculator(TheoryVectorCalculator):
    name = "shear_shear"
    def __init__(self, model, source1_name, source2_name):
        super().__init__(model)
        self.source1_name = source1_name
        self.source2_name = source2_name

    def compute(self, inputs, parameters):
        sys = self.model.calculation_systematics()
        source1 = parameters[self.source1_name]
        source2 = parameters[self.source2_name]


        outputs = ShearShearCellVector(n1, n2, source1, source2)
        # ...
        # do the Limber integral
        # ...
        for b1,b2 in bin_pairs:
            c_ell = limber_calculation(sys, source1, source2, b1, b2)
            outputs.add(b1, b2, theta, c_ell)  #need theta in __init__

        return outputs

class MultiplicativeShearBias(OutputSystematic):
    def __init__(self, source_name):
        self.source_name = source_name

    def __call__(self, name, outputs, parameters):
        if name=='shear_shear':
            self.shear_shear(outputs, parameters)
        elif name=='ggl':
            self.ggl(outputs, parameters)
        else:
            pass

    def shear_shear(self, outputs, parameters):
        """Modify the outputs from a ShearShearCalculator"""
        n1 = outputs['nbin_1']
        n2 = outputs['nbin_2']

        if outputs.source1_name == self.source_name:
            for i in range(n1):
                m_i = parameters['{}.m_{}'.format(self.source_name, i)]
                for j in range(n2):
                    if (i,j) not in outputs: continue
                    outputs[i,j] *= (1+m_i)

        if outputs.source2_name == self.source_name:
            for i in range(n1):
                for j in range(n2):
                    m_j = parameters['{}.m_{}'.format(self.source_name, j)]
                    if (i,j) not in outputs: continue
                    outputs[i,j] *= (1+m_i)

    def ggl(self, inputs, parameters):
        pass

