# Random Features

Random Fourier features (RFs) provide a finite-dimensional approximation to a kernel. 
For backbone descriptors $h$ and $h'$, Franken constructs a map
$\phi$ such that

$$
k(h,h') \approx \phi(h)^\mathrm{T}\phi(h').
$$

The model is therefore linear in the RF coefficients while retaining the nonlinear
similarity measure of a kernel method. The number of RFs controls how accurately the
finite feature map approximates the kernel and determines the number of fitted
coefficients.

## How to choose RFs?

Increasing the number of RFs improves the kernel approximation and can increase the
accuracy of the resulting potential, at the cost of additional memory and computation.
As illustrated below, a larger RF representation can attain a lower error, but requires
more training structures before the training and validation errors converge. In other
words, model capacity and sample complexity increase together.

The RF count should therefore be selected according to the training-set size. Small
datasets containing only tens of structures generally require fewer RFs, whereas larger
datasets can support more expressive models without compromising generalization.

```{figure} ../_static/TM23Cu_sample_complexity.png
:class: rounded-image
:width: 75%
:align: center

Force-error learning curves for Cu from the TM23 dataset using a MACE-MP0 backbone.
Larger RF models reach lower errors when sufficient training structures are available,
but require more samples to realize this advantage.
```

The appropriate value is system- and dataset-dependent and should be checked on a
validation set. See the {doc}`training tutorial <../notebooks/training>` for the
corresponding configuration option.

## Multiscale approach

For a Gaussian kernel, the length scale $\sigma$ controls locality in the normalized
backbone-feature space and therefore strongly affects the RF representation. A grid
search over $\sigma$ is comparatively expensive because changing it requires
recomputing the random features and their derivatives.

For this reason, Franken implements a **multiscale strategy** in which the RFs cover
several resolutions within one representation. Given a total budget of $D$ features
and $d$ length scales $\{\sigma_1,\ldots,\sigma_d\}$ sampled uniformly over a chosen
interval, Franken assigns $D/d$ features to each scale and concatenates the resulting
feature maps:

$$
\phi(h) = [\,\phi_{\sigma_1}(h)\mid\phi_{\sigma_2}(h)\mid\cdots\mid
\phi_{\sigma_d}(h)\,].
$$

Each $\phi_{\sigma_j}$ is a random Fourier feature map constructed with length scale
$\sigma_j$.

This construction captures similarities at several resolutions without fitting a
separate model for every candidate length scale. In the reported benchmarks, it
achieved accuracy comparable to an optimally tuned single-scale model while reducing
the tuning effort. See the
[Franken paper](https://doi.org/10.1038/s41524-025-01779-z) for the derivation and
benchmark results.
