import numpy as np

# Use the metadata to 
class TheoryResults:
    def __init__(self, metadata):
        self.metadata = metadata
        self.indices, self.length = self.build_indices(metadata)
        self.results = {}
        self.vector = np.zeros(self.length)

    def data_vector(self):
        return self.vector

    def build_indices(self, metadata):
        indices = {}
        i = 0
        for block in metadata['ordering']:
            name = block['name']
            n = len(block['xs'])
            indices[name] = (i, i+n)
            i+=n
        return indices, i

    def set(self, name, x):
        start, end = self.indices[name]
        self.vector[start:end] = x

    def get(self, name):
        start, end = self.indices[name]
        return self.vector[start:end]

    def to_cosmosis_block(self, block):
        pass
