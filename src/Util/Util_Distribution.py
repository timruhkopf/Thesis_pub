import torch
import torch.distributions as td
from torch.distributions import constraints


class InverseGamma(td.Gamma):

    def __init__(self, conentration, rate):
        # since IG is not available in pytorch, use equivalence:
        # if X ~ Ga(a, b), then Y = 1 / X ~ IG(a, 1/b), which in turn means,
        # to specify an IG(c,d) distrib. variable, specify one with Ga(c, 1/d)
        # https://docs.google.com/viewer?url=https%3A%2F%2Fwww.johndcook.com%2Finverse_gamma.pdf&embedded=true&chrome=false&dov=1
        super().__init__(conentration, rate ** -1)


class SpikeNSlab_mixture:
    """abstract class for SpikeNSlab """

    def __init__(self, pi, spike=None, slab=None):
        """
        p(shrunk| delta_j, ...) = (1-delta_j) spike + delta_j slab

        :param pi: probability of Binomial distribution
        :param spike: torch.distribution object, representing the dirac mass approximation
        :param slab: torch.distribution object, representing the default distribution
        """
        self.dist = {
            'spike': spike,
            'slab': slab,
            'indicator': td.Binomial(1, probs=pi)}  # can be extended to multi groups

    def sample(self):
        """returns tuple: (shrunk, delta)"""
        delta = self.dist['indicator'].sample()
        return (self.dist['spike'].sample(), self.dist['slab'].sample())[int(delta)], delta

    def log_prob(self, shrunk, delta):
        """
        Joint log prob of p(shrunk | delta_j, ...) and p(delta)
        Notice, since  delta = [0, 1],
        p(shrunk| delta_j, ...) = (1-delta_j) spike + delta_j slab
        the log prob of shrunk conditional on delta decomposes to the log prob
        of spike or slab, depending on the state
        """
        value = (self.dist['spike'].log_prob(shrunk), self.dist['slab'].log_prob(shrunk))[int(delta)]
        return sum(value + self.dist['indicator'].log_prob(delta))

    def make_continous(self):
        """
        for available transformations of transformed distributions see
        https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.transforms

        The Issue addressed here is, that delta is not continous and thus gradient
        based trajectories may yield deltas different from 0 and 1:
        see this paper on binary variables in HMC & spikeNslab as example!!
        Particullarily look at spike N slab for positive coefficients -
        which is the case of IG mixture!!!
        https://arxiv.org/pdf/1311.2166.pdf

        # be carefull with the next source, as they do GIBBS SAMPLING, soo this is
        # not particularly helpfull, BUT an intresting note on IG & student-T
        https://arxiv.org/pdf/1812.07259.pdf (on dirac & close to point.
        also notice: IG mixture is assigning marginal t distributions for
        beta's spikeNslab.
        for a potential implementation see https://pytorch.org/docs/master/distributions.html?highlight=transformeddistribution#studentt"""
        # TODO make continous spikeNslab
        pass


class SpikeNSlab_N(SpikeNSlab_mixture):
    def __init__(self, dim, pi, v0=0.001, tau=1.):
        """
        Normal mixture default (spike & slab both None) of
        p(beta_j| delta_j, tau²) = (1-delta_j) N(0, v_0 tau²) + delta_j N(0,tau²)
        delta_j | theta ~iid B(1, pi)
        v0 chosen very close to zero
        with j indicating the parameter group j=1,..., G in
        \beta = (beta_1, ..., beta_G)

        Current Implementation assumes only one Group of parameters beta_j

        :param pi: probability of Binomial distribution
        :param v0: dirac scaling factor. choose close to zero
        :param tau: variance of the slab
        """

        super().__init__(pi, spike=td.Normal(torch.zeros(dim), v0 * tau),
                         slab=td.Normal(torch.zeros(dim), tau))


class SpikeNSlab_IG(SpikeNSlab_mixture):
    def __init__(self, pi, a, b, v0):
        """
        Spike and Slab mixture of InverseGamma distributions
        The observation, that beta ~ MVN(0, tau²*I_(m_g)) can be shrunken towards zero
        by adjusting tau² leads to the following prior formulation:

        p(tau²| delta_j, a, b) = (1-delta_j) IG(a, v0 * b) + delta_j IG(a, b)
        delta_j | theta ~iid B(1, pi)

        Current Implementation assumes only one Group of parameters beta_j;
        thereby, tau² is a scalar.

        :param pi: probability of Binomial distribution
        :param a: a of IG(a, b)
        :param b: b of IG(a, b)
        :param v0: dirac scaling factor. choose close to zero
        # TODO CONSIDER TRANSFORMED DISTRIBUTIONS TO UNCONSTRAIN THE SPACE!
        """
        super().__init__(pi, spike=InverseGamma(a, v0 * b),
                         slab=InverseGamma(a, b))


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



    n = SpikeNSlab_N(dim=10, pi=0.5, v0=0.01, tau=1.)
    n.log_prob(shrunk=torch.zeros(10), delta=torch.tensor(0.))

    print(n.log_prob(*n.sample()))

    ig = SpikeNSlab_IG(pi=0.5, a=0.1, b=0.1, v0=0.01)
    n.sample()
    print(n.log_prob(shrunk=torch.zeros(10), delta=torch.tensor(1.)),
          n.log_prob(shrunk=torch.zeros(10), delta=torch.tensor(0.)))
