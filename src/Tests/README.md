# Testing Strategy

## Hidden Testing

* Figure out that the frequentistic module components work as expected
* dynamicly computed piror.log_prob works

## Convergence Testing

Convergence Testing is BAD Practice in general (as randomness is present)
and prevents automated testing (since they may fail). Consider this section rather experiments, which are designed to
work generally & on repeated execution (2-3 times or less)
should always work at least once!

1. knowing Hidden module works, try out a simple regression example with bias & one variable on MSE loss to ensure
   frequentistic optimisation works (the module works fine)

2. knowing (1) works, check if frequentistic optimisation can find the modulo of the log_prob i.e. SGD on log_prob is
   sensible on the simple regression example

3. try out fully bayesian regression.

From there on out, check the modules (such as GAM):

1. check the additional functionallity of the module
2. in a very simple (!!!) but meaningful example try out a sampling procedure, that is almost guaranteed to succeed.
   Make use of Test_Samplers' Convergence_teardown. The latter checks if the sampler progressed away from init and in
   terms of avg_MSE, if the model was improved significantly towards the true model.
   
BE AWARE, that complex models such as ShrinkageBNN are likely to fail due to extreme initialisations from 
the prior. Make sure, to look at the general trend of the models (and repeated runs if necessary) to figure out 
if the models manage to learn sth. meaningful.
To make meaningful tests in these scenarios use explicit functions to generate data from them
and estimate using the ShrinkageBNN model.
