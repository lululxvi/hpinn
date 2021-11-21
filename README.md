# hPINN: Physics-informed neural networks with hard constraints

The source code for the paper [L. Lu, R. Pestourie, W. Yao, Z. Wang, F. Verdugo, & S. G. Johnson. Physics-informed neural networks with hard constraints for inverse design. *SIAM Journal on Scientific Computing*, 43(6), B1105-B1132, 2021](https://doi.org/10.1137/21M1397908).

## Code

The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v0.9.1. If you want to use the latest DeepXDE, you need to modify the code.

### Holography

- [hPINN](holography)
- FDFD: [Jupyter Notebook](FDFD/inverse_design_FDFD-epsstart-eps1.ipynb)
- FEM: [Jupyter Notebook](FEM/Main.ipynb)

### Fluids in Stokes flow

- [hPINN](stokes/stokes.py)

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{lu2021physics,
  author  = {Lu, Lu and Pestourie, Raphael and Yao, Wenjie and Wang, Zhicheng and Verdugo, Francesc and Johnson, Steven G},
  title   = {Physics-informed neural networks with hard constraints for inverse design},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {43},
  number  = {6},
  pages   = {B1105-B1132},
  year    = {2021},
  doi     = {10.1137/21M1397908}
}
```

## Questions

To get help on how to use the code, simply open an issue in the GitHub "Issues" section.
