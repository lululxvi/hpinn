# hPINN: Physics-informed neural networks with hard constraints

The source code for the paper [L. Lu, R. Pestourie, W. Yao, Z. Wang, F. Verdugo, & S. G. Johnson. Physics-informed neural networks with hard constraints for inverse design. *arXiv preprint arXiv:2102.04626*, 2021](https://arxiv.org/abs/2102.04626).

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
  title   = {Physics-informed neural networks with hard constraints for inverse design},
  author  = {Lu, Lu and Pestourie, Raphael and Yao, Wenjie and Wang, Zhicheng and Verdugo, Francesc and Johnson, Steven G},
  journal = {arXiv preprint arXiv:2102.04626},
  year    = {2021}
}
```

## Questions

To get help on how to use the code, simply open an issue in the GitHub "Issues" section.
