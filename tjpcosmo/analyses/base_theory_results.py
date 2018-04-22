
class TheoryResults:
    slots = []

    def to_cosmosis_block(self, block):
        for key in self.slots:
            block['results', key] = getattr(self, key)