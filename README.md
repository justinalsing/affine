# affine

Parallelized affine invariant MCMC sampling implemented in `jax`. 

You must install `jax` yourself due to `jax` not installing `CUDA` and `CuDNN` by default.

For use with CPU devices

```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

then

```pip install git+https://github.com/justinalsing/affine.git```

If you would like to use a GPU accelerator, simply install `jax` as required and 
then 

```pip install git+https://github.com/justinalsing/affine.git```

Usage see `affine_sampler_example.ipynb`.
