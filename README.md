# difflogic-iwp: Scalable, stable and slim difflogic 
## ⚙️ Setup the environment
1. ⏬ **Clone the Repository**  
   Use the following command to clone this repository:
   ```bash
   git clone https://github.com/anonymous-user/difflogic-iwp.git
   ```

2. 🏗️ **Build the container environment**   
   Make sure that [apptainer](https://apptainer.org/docs/admin/main/installation.html#) is installed ([singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) works too).
   Then, build this minimal container environment.

   ```bash
   apptainer build ./execution_setup/minimal.sif ./execution_setup/minimal.def
   ```

## 🧑‍🔬 Reproduce the experiments
Execute the following command to train a reparametrized LGN on CIFAR-100.
```bash
apptainer exec ./execution_setup/minimal.sif python main.py --architecture "lgn" --dataset "cifar-100" --iwp --weights_init "ri"
```

See `reproduce.md` for the exact commands used in each experiment.

### Adjust data and results directories
The file `./execution_setup/directories.py` contains the directories to store datasets and logging results. They are set to default values initially, but you can change them manually.

## 🧰 Extend the codebase
To extend the implementation yourself, we recommend to use the extended container environment `complete.def`, with installs PyTorch development tools instead of only the runtime.
```bash
sudo apptainer build ./execution_setup/complete.sif ./execution_setup/complete.def 
```
This allows you to recompile the CUDA implementation.
