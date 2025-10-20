# SVD_by_LFA
Singular value decomposition of convolutional layer by local Fourier analysis.

## Description
This repository provides a framework to efficiently calculate the singular values (SVD) (and if needed singular vectors) of 
convolutional layers via local Fourier analysis (LFA). While the explicit calculation of the SVD of convolutional layers
is infeasible and computationally limited since the corresponding weight matrix grows quickly for increasing input dimensions. 
Transferring the weight tensor into the Fourier domain allows for efficient SVD computation. Note, that this requires periodic
boundary conditions (periodic padding) instead of Dirichlet boundary conditions (zero padding). 
For growing input dimensions the influence of boundary conditions becomes negligible (see Fig. below).


For more details see https://arxiv.org/pdf/2506.05617

![Effect of boundary conditions for increasing input sizes (n = 4, 8, 32; left to right). The number of input and output
channels is fixed to 16](ims/boundary_cond.png)
Effect of boundary conditions for increasing input sizes (n = 4, 8, 32; left to right). The number of input and output
channels is fixed to 16.


## Requirements
Running with Python3.6 and CUDA Version: 11.0

## Usage
Currently supported methods: local Fourier analysis ('lfa') or explicit ('expl').  
```bash
python3 main.py --method 'lfa'
```


