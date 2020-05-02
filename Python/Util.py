import inspect


def setkwargs(method):
    """method decorator to attach the methods args, kwargs and, if not specified,
     default kwargs to instance as attribute"""

    def wrapper(self, *args, **kwargs):
        # finding all arg names without self
        names = inspect.getfullargspec(method).args[1:]
        names = [n for n in names if n not in kwargs.keys()]

        # default arguments:
        defaults = {
            k: v.default
            for k, v in inspect.signature(method).parameters.items()
            if v.default is not inspect.Parameter.empty}

        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)
        self.__dict__.update({k: v for k, v in zip(names, args)})

        method(self, *args, **kwargs)

    return wrapper
