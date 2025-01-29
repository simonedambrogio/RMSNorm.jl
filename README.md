# RMSNorm.jl

A Lux.jl neural network layer implementing Root Mean Square Layer Normalization (RMSNorm) as described in ["Root Mean Square Layer Normalization"](https://arxiv.org/abs/1910.07467) by Zhang and Sennrich (2019).

## Overview

RMSNorm is a simplified variant of Layer Normalization that only normalizes by the root mean square, without centering. This makes it computationally more efficient while maintaining similar performance to LayerNorm in many applications. This implementation is built as a layer for the [Lux.jl](https://github.com/LuxDL/Lux.jl) deep learning framework.

The normalization is computed as:

```math
y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} * \gamma
```

where γ is an optional learnable scale parameter.

## Installation

```julia
using Pkg
Pkg.add("RMSNorm")
```

## Usage

```julia
using RMSNorm
using Random
using Lux

# Create an RMSNorm layer
rng = Random.default_rng()
input_size = (768,)  # Feature dimension
layer = RMSNorm(input_size)

# Initialize parameters
ps = Lux.initialparameters(rng, layer)

# Example input: batch of 32 samples
x = randn(Float32, 768, 32)
y, st = layer(x, ps, NamedTuple())
```

### Constructor Options

```julia
RMSNorm(shape; 
    activation=identity,     # Activation function to apply after normalization
    epsilon=1f-5,           # Small constant for numerical stability
    dims=Colon(),           # Dimensions to normalize over
    affine=true,            # Whether to use learnable scale parameter
    init_scale=ones32       # Initialization function for scale parameter
)
```

## Features

- Full integration with the Lux.jl deep learning framework
- Efficient implementation using Julia's broadcasting
- Support for custom activation functions
- Optional learnable scale parameter
- Configurable normalization dimensions
- Type-stable implementation

## Citation

If you use this package in your research, please cite:

```bibtex
@article{zhang2019root,
  title={Root mean square layer normalization},
  author={Zhang, Biao and Sennrich, Rico},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
