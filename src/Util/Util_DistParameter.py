from torch.nn import Parameter

class StateBij:
    def sample(self):
        pass

    def log_prob(self):
        pass

    @property
    def tensor(self):
        # TODO return the unbij value
        return self.context.data

    @tensor.setter
    def tensor(self, value):
        # TODO take a value & transfrom it before setting (notice, that .data is unaffected so
        #  the usual settings e.g. during sampling is unaffected)
        self.context.data = value


class StateUnbij:
    def sample(self):
        pass

    def log_prob(self):
        pass

    @property
    def tensor(self):
        return self.context.data

    @tensor.setter
    def tensor(self, value):
        self.context.data = value



class DistParameter(Parameter):

    def __init__(self, tensor, dist, transform=None, *args, **kwargs):
        super().__init__(self, tensor, *args, **kwargs)
        self.dist = dist

        if transform is None:
            self.state = StateUnbij()
        else:
            self.state = StateBij()

    @property
    def state(self, State):
        self._state = State
        self._state.context = self

    def sample(self, *args, **kwargs):
        self._state.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        self._state.log_prob(*args, **kwargs)

    @property
    def tensor(self):
        return self._state.tensor

    @tensor.setter
    def tensor(self, value):
        self._state.tensor = value

if __name__ == '__main__':
    pass
