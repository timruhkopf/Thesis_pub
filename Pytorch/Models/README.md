This readme file is intended to introduce the structure on 
writing the bayesian neural networks layers in Pytorch:

## (1) Parsing & Prior Model
Due to the inability of writing on the nn.Parameters using vector_to_parameters
AND caching the gradients - i.e. doing so will fail in the samplers.
However, to reduce the amount of parsing necessary, nn.Module is used
with hamiltorchs' unflatten function (which is the not inplace version of 
aforementioned vector_to_parameters) in order to infer the list of tensors 
implied by the vector. In order to do so in the context of hierarchical models
a particular pattern must be followed:

#### (1.1) Model based
NNs typically consist of Hidden layers (or layers, that inherit from Hidden - which facilitates
parsing and the necessity to write code significantly. The idea is, that a
model is an aggregation of layers and should actually delegate most of the parsing, 
sampling & prior_evaluation to the layers themselve (and define peculiarities there). 
To do so the Model requires an attribute self.layers such as:

    self.layers = nn.Sequential(self.shrinkage[shrinkage](hunits[0], hunits[1], True, activation),
            *[layer.Hidden(no_in, no_units, True, activation)
              for no_in, no_units in    zip(self.hunits[1:-2], self.hunits[2:-1])],
            layer.Hidden(hunits[-2], hunits[-1], bias=False, activation=final_activation))

this facilitates e.g. the prior_log_prob formulation to 

    p_log_prob = sum([h.prior_log_prob().sum() for h in self.layers])

the only required "hand" parsing is done in vec_to_attrs (code taken from BNN)
is to split the received vector in sub vectors, each characterising an entire layer.
to do this the number of parameters of each layer must be known. This is 
facilitated by a property Hidden.n_params.

    lengths = list(accumulate([0] +[h.n_params for h in self.layers]))
    for i, j, h in zip(lengths, lengths[1:], self.layers):
            h.vec_to_attrs(vec[i:j])    

    # CONSIDER ADDITIONAL NON LAYER RELATED PARAMETERS ENCODED IN THE VEC
    # HERE: a variance parameter of the log-likelihood
    if self.heteroscedast: 
            self.__setattr__('sigma', vec[-1])   

#### (1.2) Layer based design
To enable nn.Module based parsing each parameter has to be set thrice:
 
    self.W_ = nn.Parameter()
    self.W = torch.tensor()
    # notice the vector valued distribution
    self.dist['W'] = td.Normal(torch.zeros(p_x * p_h), 1.) 
 
 where the trailing underscore indicates the nn.Parameter by convention.
 The attribute with no trailing underscore is the attribute which is actually
 used for evaluation in "forward", "reset_parameters" and "prior_log_prob".
 at least once the nn.Parameter must be filled with a parameter of the actual desired shape,
 to leverage the parsing of nn.Module e.g. 
 
    # notice the reshape to matrix
    self.W = self.dist['W'].sample().view(p_x, p_h)
    self.W_.data = self.W
    
care must be taken if the model is a hierarchical one, 
in which case it is crucial to update the distributional parameters, before 
sampling & prior_log_prob evaluations:

**vec_to_attrs()**
this function takes a vector as input and zips the named parameters (with no
underscore) with the appropriately (infered by nn.Module) list of tensors, 
to set the attribute (e.g. self.W).
If no hierarchical structure is required, Hidden.vec_to_attrs should be added 
to the classes methods. Otherwise consider Group_lasso.vec_to_attrs, as it 
already uses the vecs parameters and calls update_distributions.
    
    tensorlist = hamiltorch.util.flatten(self, vec)
    # Note that p_names is property attrib. of Hidden class
    for name, tensor in zip(self.p_names, tensorlist):
        self.__setattr__(name, tensor)

**update_distribution()**
this function is used during vec_to_attrs in order to update the distribution
based on the vec proposals and maintain the correct prior distributions for
prior_log_prob evaluations & reset_of_parameters. e.g. after calling vec_to_attrs

    self.dist['W_shrinked'].scale = self.tau

**prior_log_prob()**
should the prior model exhibit a more complex structure
for a single parameter, this is the place to explicitly encode this information
e.g.  a distributional assumption on W, that the first column; i.e. all weights
attached to the first variable in X follows another distribution dependent on self.tau
can be encoded in the following way:

     self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_in), self.tau)
     self.dist['W'] = td.Normal(torch.zeros(self.no_in * (self.no_out - 1)), 1.)
     
     self.W = torch.cat([self.dist['W_shrinked'].sample().view(self.no_in, 1),
                            self.dist['W'].sample().view(self.no_in, self.no_out - 1)],
                           dim=1)
                           
The resulting prior_log_prob can be encoded as:
    
    param_names = self.p_names 
    param_names.remove('W')
    
    value = torch.tensor(0.)
    # evaluate all other nn.Parameters in their respective distributions. 
    # be carefull with matrix valued parameters, to properly shape them!
    for name in param_names:
        value += self.dist[name].log_prob(self.__getattribute__(name)).sum()

    # W's split & vectorized priors
    value += self.dist['W_shrinked'].log_prob(self.W[:, 0]).sum() + \
             self.dist['W'].log_prob(self.W[:, 1:].reshape(p_x * (p_h - 1))).sum()


**reset_parameters()**
is the sampling method to initialize the parameters. Here the hierarchical 
structure of the sampling process can be encoded:

    self.tau = self.dist['tau'].sample()
    self.dist['W_shrinked'].scale = self.tau # this is MANDATORY!
    self.W = torch.cat([self.dist['W_shrinked'].sample().view(self.no_in, 1),
                        self.dist['W'].sample().view(self.no_in, self.no_out - 1)],
                       dim=1)

typically, you would want to call this function at the end of init
or when sampling a prior data model.

## (2) Posterior log_prob

#### (2.1) SG- & full dataset log_prob
Using MCMC Samplers requires the evaluation of the log posterior probability,
which is p(\theta | Y, X) = p(Y | \theta) p(\theta). Using the following structure
allows to formulate a (**stochastic gradient capable**) log_prob model:

    def likelihood(self, X):
        """:returns the conditional distribution of y | X"""
        # notice self(X) is self.__call__(X) and defines the forward path
        return td.Normal(self(X), scale=self.sigma)
 
    def log_prob(self, X, y, vec):
        """SG flavour of Log-prob: any batches of X & y can be used"""
        self.vec_to_attrs(vec)  # parsing to attributes
        return self.prior_log_prob() + \
               self.likelihood(X).log_prob(y).sum()

To make the model not only applicable to Stochastic Gradient flavours of MCMC, but 
ready it to handle fixed and **full dataset** samplers simply use this closure,
which in place changes the self.log_prob to the partial function, entailing your
"default" arguments for X and y:

     def closure_log_prob(self, X=None, y=None):
         """
         log_prob factory, to fix X & y for samplers operating on the entire dataset.
         :returns None. changes inplace attribute log_prob
         """
         print('Setting up "Full Dataset" mode')
         self.log_prob = partial(self.log_prob, X, y)

#### (2.2) Layers as Unittest regression models 
An interesting observation for testing both your layers and your samplers is,
that in fact the layers are a prior model themselves may correspond to 
the forward path modeling e.g. the location of a Normal likelihood. 
providing a layer with a likelihood & log_prob makes them complete models 
themselves. Due to the consistent (strong) naming conventions of the 
methods, the likelihood & log_prob fit easily, **making the layers stand alone 
models**. The bayesian formulation of the prior model allows sampling data from 
the prior & likelihood:
    
    # BNN EXAMPLE:
    bnn = BNN(hunits=[1,10,5,1]) # assuming init also initializes weights & biases
    # generate data
    X_dist = td.Uniform(torch.tensor(-10.), torch.tensor(10.))
    X = X_dist.sample(torch.Size([100])).view(100, 1)
    y = bnn.likelihood(X).sample()
    
    # LAYER REGRESSION MODEL:
    no_in = 2
    no_out = 10
    reg = Hidden(no_in, no_out, bias=True, activation=nn.Identity())
    # init calls reset_parameters, sampling a prior data model (true model)
    
    X_dist = td.Uniform(torch.tensor([-10., -10]), torch.tensor([10., 10]))
    X = X_dist.sample(torch.Size([100]))
    y = reg.likelihood(X).sample() # since self.likelihood returns a td.Distribution object

    

