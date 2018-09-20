likelihood_registry = {}


class BaseLikelihood:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.name if hasattr(cls, 'name') else cls.__name__
        name = name.lower()
        likelihood_registry[name] = cls

    @staticmethod
    def from_name(name):
        return likelihood_registry[name.lower()]

    def __init__(self, data):
        self.data = data
