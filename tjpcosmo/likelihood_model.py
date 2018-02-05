"""
These are very thin cosmosis wrappers that connect to tell it how to connect
to the primary TJPCosmo code.

"""

def setup(options):
    return {}

def execute(block, config):
    block['likelihoods', 'lsst_like'] = 0.0
    return 0

def cleanup(config):
    pass
