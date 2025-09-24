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

### 📁 Adjust data and results directories
The file `./execution_setup/directories.py` contains the directories to store datasets and logging results. They are set to default values initially, but you can change them manually.

### 📊 Generate your own plots
We provide the python scripts that we used to generate the plots in our paper.
To run those, you first need to build the extended container environment `complete.def`, which additionally installs a LaTeX environment along with Python packages for plotting.
```bash
sudo apptainer build ./execution_setup/complete.sif ./execution_setup/complete.def 
```
See `reproduce.md` for instructions on the script usage. 

## 🧰 Extend the codebase
To extend the implementation yourself, have a look at `repository.md` for an overview on the codebase structure.
