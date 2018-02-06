
class TheoryResults:
    slots = []

    def to_cosmosis_block(self, block):
        for key in self.slots:
            block['cosmological_parameters', key] = getattr(self, key)