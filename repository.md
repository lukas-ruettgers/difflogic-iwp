# Codebase structure

## Differentiable logic gate networks
```
difflogic/                             # Implementation of logic gate networks
│
├── cuda/                              # CUDA implementation of reparametrization
│   └── difflogic_iwp_backward_w.cu    # Backward kernel for partial derivatives w.r.t local weights
│   └── difflogic_iwp_backward_x.cu    # Backward kernel for partial derivatives w.r.t inputs
│   └── difflogic_iwp_forward_train.cu # Forward kernel for continuous relaxation
│   └── difflogic_iwp_forward_eval.cu  # Forward kernel for discretized logic gate circuit 
│   └── difflogic_iwp_shared.cuh       # Header file with shared macros 
│
├── difflogic.py                       # Python wrapper class for reparametrization 
├── functional.py                      # Custom autograd functions 
```
### Recompile CUDA implementation
If you make changes to the CUDA implementation and want to recompile it, run
   ```bash
   apptainer ./execution_setup/minimal.sif python setup.py build_ext --inplace --force
   ```

## Experiment infrastructure
```
library/                               
│
├── config.py                          # Experiment parameters
│
├── datasets.py                        # Dataset preprocessing and loading
│
├── models.py                          # Model definitions
│
├── train.py                           # Model training 
│
├── measurements.py                    # Measurements 
```
