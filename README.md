# SVD_by_LFA
## Singular value decomposition of convolutional layers via local Fourier analysis.

## Description
This repository provides an eficient framework for computing the singular values (SVD) and optionally the singular vectors of 
convolutional layers using local Fourier analysis (LFA). 

The direct calculation of the SVD of a convolutional layer is typically infeasible because the corresponding weight matrix grows rapidly with increasing input dimensions. 
By transferring the convolutional weight tensor into the Fourier domain, this Framework enables efficient and scalabel SVD computation. Note, that this requires periodic
boundary conditions (periodic padding) instead of Dirichlet boundary conditions (zero padding). 
For large input dimensions the influence of boundary conditions becomes negligible (see Figure below).


![Effect of boundary conditions for increasing input sizes (n = 4, 8, 32; left to right). The number of input and output
channels is fixed to 16](ims/boundary_cond.png)
Effect of boundary conditions for increasing input sizes (n = 4, 8, 32; left to right). The number of input and output
channels is fixed to 16.

For theoretical background and derivations visit DOI 10.3233/FAIA250793

## Requirements
- Python3.6 
- CUDA Version: 11.0
- NumPy 
- PyTorch

Install dependencies via:
```bash 
pip install -r requirements.txt
```

## Usage
Two methods are currently supported
- `lfa` local Fourier analysis for large-scale, efficient SVD computation 
- `explicit`  Explicit SVD computation for small-scale verification

Example usage: 

```bash
python3 main.py --method lfa
```

or 

```bash
python3 main.py --method expl
```

## Citation 

If you use this code in your research, please cite:


@inproceedings{vanBetteray2025LFA,  
  author    = {Antonia van Betteray and Matthias Rottmann and Karsten Kahl},  
  title     = {LFA Applied to CNNs: Efficient Singular Value Decomposition of Convolutional Mappings by Local Fourier Analysis},  
  booktitle = {Frontiers in Artificial Intelligence and Applications, Volume 413: ECAI 2025},  
  pages     = {90--97},  
  year      = {2025},  
  doi       = {10.3233/FAIA250793},  
  publisher = {IOS Press}
}




