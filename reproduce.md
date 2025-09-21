# Reproduce experiments
This file contains the specific parameters that we used in our experiments.

## Evaluation
### Logic Gate Networks
To train a logic gate network, run
```bash
apptainer exec ./execution_setup/minimal.sif python main.py -a "lgn" -d "cifar-100" -s SEED --depth_scale DEPTH_SCALE --iwp --weights_init "ri"
apptainer exec ./execution_setup/minimal.sif python main.py -a "lgn" -d "cifar-100" -s SEED --depth_scale DEPTH_SCALE --op --weights_init "ri" --wandb_verbose "eval"
```

| Parameter                 | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `SEED`                  | Random seed. See `./library/config.py` for the values used.                |
| `DEPTH_SCALE`           | Factor by which the model is scaled in depth (e.g. `1`, `2`, `3`, `5`, `20`). |
| `--iwp` / `--op`          | Use the IWP (reparametrized) or OP (original parametrization) model.           |
| `--wandb_verbose eval`    | Log evaluation metrics to [Weights & Biases](https://wandb.ai).            |


### CNN baseline
To train the CNN baseline architecture, execute
```bash
apptainer exec ./execution_setup/minimal.sif python main.py -a "cnn" -d "cifar-100" --wandb_verbose "eval" -lr 0.001
```

| Parameter                 | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `-a "cnn"`                  | CNN architecture.                |
| `-lr 0.001`                  | Adjusted learning rate.                |
| `--wandb_verbose eval`    | Log evaluation metrics to [Weights & Biases](https://wandb.ai).            |



## Timing experiments
```bash
apptainer exec ./execution_setup/minimal.sif python main.py -a "lgn" -d "cifar-100" -s SEED P --log_verbose "timing" --n_timing_measurements 20 --weights_init "ri" --depth_scale 20 -ni 0
```

| Parameter                 | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `SEED`                  | Random seed. See `./library/config.py` for the values used.                |
| `P`           | Use either `--iwp` or  `--op` here to specify the parametrization. |
| `--n_timing_measurements 20"` | Number of timing measurements. |

## Intermediate gate output distribution
### Out-of-the-box initialization schemes
```bash
apptainer exec ./execution_setup/minimal.sif python main.py -a "lgn" -d "cifar-100" -s SEED P --log_verbose "features" --weights_init WI --depth_scale 40 --log_timestamp T -ni NI
```

| Parameter                 | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `SEED`                  | Random seed. See `./library/config.py` for the values used.                |
| `P`           | Use either `--iwp` or  `--op` here to specify the parametrization. |
| `WI` | Which weight initialization to use, e.g. `gauss`, `ri`, `and-or`, `uniform`. |
| `T` | Log the feature distribution after `T` batches, e.g. `0`,`100`,`1000`. |
| `NI` | Make sure that the number of iterations is larger than `T`. |


## Gradient analysis
```bash
apptainer exec ./execution_setup/minimal.sif python main.py -a "lgn" -d "cifar-100" -s SEED P --log_verbose "features" --weights_init WI --depth_scale 40 --log_timestamp T -ni NI
```
| Parameter                 | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `SEED`                  | Random seed. See `./library/config.py` for the values used.                |
| `P`           | Use either `--iwp` or  `--op` here to specify the parametrization. |
| `WI` | Which weight initialization to use, e.g. `gauss`, `ri`, `and-or`, `uniform`. |
| `T` | Log the feature distribution after `T` batches, e.g. `0`,`100`,`1000`. |
| `NI` | Make sure that the number of iterations is larger than `T`. |


## Custom heavy-tail weight initializations
There are multiple parameters to customize:
| Parameter                 | Description | Example Values                                                                 |
|---------------------------|-----------------------------------------------------------------------------| ------------------------------------|
| `--weights_init` | Which weight initialization to use | `gauss`, `ri`, `and-or`, `uniform`. |
| `--sigma` | Standard deviation $\sigma$ | `0.25` (IWP default), `1.0`, `4.0`
| `--init_shift` | Heavy-tail shift $\pm \mu$ | `1.2` (default for sinusoidal), `1.5`
| `--init_shift_direction` | Direction to shift the four output tuples to | `0110` (see [below](#init-shift-direction))

#### Init Shift Direction
The four binary values XXXX specify the outputs of the boolean function that the heavy-tail initialization shall bias towards.
The output tuples are specified in lexicographic order: $(0,0), (0,1), (1,0), (1,1)$.
A few examples below:
| Initialization                 | Value                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Residual initialization (default) | `0101` (or `0011`) |     
| XOR | `0110` |     
| OR | `0111` |     
| AND | `0001` |  

## Regularizations
| Regularization method | Add to the parameter list of the command |     
|---------------------------|-----------------------------------------------------------------------------|
| Residual connections | `--resconnect` |     
| Dropout | `--dropout_prob P`, e.g. P=`0.05`,`0.1`.  |     
| Random gate interventions | `--random_outage RO --random_outage_prob P`, where P as above, and `RO` admits `const0`, `const1`, `const0.5`, `uniform`, `bernoulli`. |    

## Binary encoding
The binary encoding of the channel values is specified via `--encoding ENC`, where ENC can take the following values
| Encoding method | Parameter value ENC |     
|---------------------------|-----------------------------------------------------------------------------|
| Raw real input | `real-input` |
| Thermometer encoding with 3 thresolds | `3-thresholds` |
| Thermometer encoding with 7 thresolds | `7-thresholds` |
| Thermometer encoding with 15 thresolds | `15-thresholds` |
| Thermometer encoding with 23 thresolds | `23-thresholds` |
| Thermometer encoding with 31 thresolds | `31-thresholds` |
