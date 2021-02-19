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

The most tedious and cumbersome is the non-naive update process of the hierarchical distributions. Ideally, dependent
parameters of the distribution would be directly linked to the underlying nn.Parameter (which indicates a graph), but
this seems not to be supported by torch.distribution. In turn, this motivates the explicit formulation of BaseModel.
update_distribution().


## Git - Workflow:
After one and a half years of developing the project, it has grown and explored ideas in many directions. 
As consequence, the code base had become somewhat tedious. Many a experiments have become legacy and
some ideas have not reached a stable version, but cluttered the repo. Moving forward from branch stable in 
git revision 7631d02d, the repo is designed in a more Software-Development fashion:

* **_stable_** branch is destined to contain only the most rigorously tested  and proven pieces of software.
  
* **_dev_** branch is a branch of **_stable_**, but contains all the Grid / BOHB and shell scripts required to 
    do the dirty work of progressing the research. In addition, more or less proven components such as e.g. GAM,
  that are not entirely stable yet reside here.
  
* **_feature_** branch. Any research idea will be a branch of **_dev_**, and ideally isolate particular lines of 
  thought, that alter or add to **_dev_**'s code.
  
Update cycle: Whenever a feature is successfully explored and deemed to contributed to the code base, it is merged 
back to **_dev_**. When rigorously tested and/or proven, this feature can be cherry picked to **_stable_** (to avoid 
moving the dirty work tools to stable).
As consequence of this, the other **_feature_** branches will require an update from **_dev_**, before being capable of 
merging into **_dev_**. This will ensure, that the ideas work on the premises they were build on (codewise) - and will 
work even if they were set aside and other ideas moved to **_dev_**. This will highlight the changes implied by the idea
and ensure state safety of the repo.
