# NaturalGradient_BBVI

BBVI/MC gradient based variational inference don't always work, and posterior collapse is one of the main obstacles. 

the hypothesis here is that the first thing any numerical methods class teaches about gradient descent is that it can get stuck in a "narrow valley", and hence justifying the use of newton method and the use of conditioners to correct the gradients; in VI, we use what is called Stochastic VI (in its original Hoffman et al flavor with natural gradient not just any gradient) to do something similar. hence i want to try and see if altering the backprop gradient to natural gradient can help.

in this repo we have a couple experiments, all with reparametrized gradient, and without amortized inference to find out what is going on:

1. LDA with reparametrized dirichlet distribution. with natural gradient in the backprop, we can recover meaningful topics where each topic has different words; backprop with just vanilla gradient will result in posterior collapse for all topics.
2. a Bayesian archetypal analysis. (WIP)
3. VAE without encoder (WIP)
