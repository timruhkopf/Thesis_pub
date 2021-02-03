# Extracting main effects from Bayesian Neural Nets using Splines and Shrinkage methods

This repo represents the entire codebase of my master thesis and the subsequent publication of the same. The main idea
is to sever out variables from the Neural Net's interaction surface, if they are purely additive in nature. To do so, a
Bayesian Neural Net, whose input layer places additional hierarchical shrinkage assumptions (on the first variable only)
, is estimated simultaneous with a spline model (of the first variable). The model's prediction is

    $\hat{y} = \hat{y}_{ShrinkageBNN(X)} +  \hat{y}_{spline(x_1)}$

To write the underlying (hierarchical) parts and models in a compact and comprehensible pythonic fashion that is also
suitable for various sampler interfaces, I developed an axiomatic class design to formulate these bayesian models. For
more on this visit the **tutorial**. Furthermore, by example, I display how bijection of constrained parameters and more
structured assumptions on the weight matrix can be implemented.

#### Implementational Note: A Workaround

The most tedious and cumbersome is the non-naive update process of the hirarchical distributions. Ideally, dependent
parameters of the distribution would be directly linked to the underlying nn.Parameter (which indicates a graph), but
this seems not to be supported by torch.distribution. In turn, this motivates the use BaseModel.update_distribution()


