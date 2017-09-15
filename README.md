### wasserstein-dist 
============================

wasserstein-dist is a tensorflow implementation of the Wasserstein 
(aka optimal transport) distance between a fixed set of data points 
and a probability distribution (from which one can sample). 
It can also be used to compute the distance between to points sets,
but it is not optimized for this purpose. 

The implementation follows the semi-dual Algorithms 2 in [Genevay Aude, 
Marco Cuturi, Gabriel Peyre, Francis Bach, "Stochastic Optimization for 
Large-scale Optimal Transport", NIPS 2016]. 

---

This is not an official Google product.
