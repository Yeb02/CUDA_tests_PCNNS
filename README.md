# Predictive coding experiments

Experiments with predictive coding and its variants, in a supervised learning setting (MNIST).  The graph's architecture will be that of a standard Multi-Layer Perceptron. Unless explicitly mentionned, both PC and iPC are tested.

### Causality, reversed order

From the [variational inference](https://arxiv.org/pdf/2202.09467.pdf) point of view, the deepest layer, the $L$-th, the one that is not predicted by any other, is a prior of a generative network, a *cause*. The generated output, layer $0$, is the most likely *consequence*. Given a pair (label, datapoint), which one should be regarded as the *cause* of the other ? Both direction of causality are easy to implement in the predictive coding framework. The case where the label is the consequence is extensively described in the [litterature](https://pubmed.ncbi.nlm.nih.gov/28333583/). Otherwise, when the label is the cause, a slight change to the implementation must be made. During the learning phase, a quadratic loss between the $L$-th layer and the target label is added to the energy subject to minimization. Then, it is as if the $L$-th layer was not the last but just an ordinary layer, whose activations are predicted to be the target label by a virtual $L+1$-th layer. The $0$-th layer can be fixed at the datapoint, totally or partially if the datapoint is corrupted and to be reconstructed.


### Order of propagation

At each inference step, activations are modified along the gradient to minimize the free energy. This modification happens simultaneously at all layers, since we consider the energy as a function of all activations. We could instead consider it as a function of only the first layer's activations, then only of the second layer's, and so on. While it decreases the potential parallelization of the algorithm, and the quality of the energy minimization, there are also advantages. The most important being that information travels much faster through the network:  in one inference step, the update at the $L$-th layer is already a function of the activations of the $0$-th layer. On the other hand, when updates are simultaneous, it takes $L$ inference steps for information to reach the other end of the network. <br>
This property may not be of much interest when studying supervised learning, but could prove useful when the network is the cognitive module of an agent in a reinforcement learning setting, where reaction time matters. <br> 
<br>
It also could be more biologicaly plausible, but I am not knowledgable enough to conclude.
