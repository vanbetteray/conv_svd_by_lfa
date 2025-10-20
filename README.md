# SVD_by_LFA
## Singular value decomposition of convolutional layer by local Fourier analysis.

## Description
This repository provides an eficient framework for computing the singular values (SVD) and optionally the singular vectors of 
convolutional layers using local Fourier analysis (LFA). 

The direct calculation of the SVD of a convolutional layer is typically infeasible because the corresponding weight matrix grows rapidly with increasing input dimensions. 
B< transferring the convolutional weight tensor into the Fourier domain, this Framework enables efficient and scalabel SVD computation. Note, that this requires periodic
boundary conditions (periodic padding) instead of Dirichlet boundary conditions (zero padding). 
For large input dimensions the influence of boundary conditions becomes negligible (see Figure below).

For theoretical background and derivations visit https://arxiv.org/pdf/2506.05617


![Effect of boundary conditions for increasing input sizes (n = 4, 8, 32; left to right). The number of input and output
channels is fixed to 16](ims/boundary_cond.png)
Effect of boundary conditions for increasing input sizes (n = 4, 8, 32; left to right). The number of input and output
channels is fixed to 16.


## Requirements
- Python3.6 
- CUDA Version: 11.0
- NumPy 
- PyTorch

```bash
pip install -r requirements.txt
```

## Usage
Two methods are currently supported
- ```bash lfa ``` local Fourier analysis for large-scale, efficient SVD computation 
- ```bash explicit```  Explicit SVD computation for small-scale verification
```bash
python3 main.py --method 'lfa'
```


