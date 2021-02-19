import torch
import torch.distributions as td
from torch.distributions import constraints


class LogTransform(td.Transform):
    r"""
    Transform via the mapping :math:`y = \log(x)`.
    allowing to sample a distribution on R+ and unconstraining it to R:
    unconst_ga = td.TransformedDistribution(td.Gamma(0.01, 0.01), LogTransform())

    Using an unconstrained distribution may help gradient based samplers
    such as HMC but requires to transform the posterior samples of the unconstrained
    parameter back to the constrained space!

        a = unconst_ga.sample([samples])
        b = unconst_ga.transforms[0]._inverse(a)
    """

    # see td.Transform for details of following attrib.
    domain = constraints.positive
    codomain = constraints.real
    bijective = True
    sign = +1  # monotone increasing function? - yes

    def __eq__(self, other):
        return isinstance(other, LogTransform)

    def _call(self, x):
        return x.log()

    def _inverse(self, y):
        return y.exp()

    def log_abs_det_jacobian(self, x, y):
        return 1 / x  # todo check it is not 1/y


if __name__ == '__main__':
    samples = 2
    seed = 1
    torch.manual_seed(seed)
    ga = td.Gamma(0.01, 0.01)
    a = ga.sample([samples])
    torch.manual_seed(seed)
    b = ga.sample([samples])

    unconst_ga = td.TransformedDistribution(td.Gamma(0.01, 0.01), LogTransform())
    torch.manual_seed(seed)
    c = unconst_ga.sample([samples])
    d = unconst_ga.transforms[0]._inverse(c)


