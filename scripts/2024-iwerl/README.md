# Scripts to perform the experiments for the 2024 IWERL paper


All of these commands are assumed to be run from within a Nix shell environment
which can be entered by doing at the root of the repository

```
nix develop --impure
```


Install Julia dependencies by opening a Julia shell (`julia --project=.`) and then doing

```
import Pkg; Pkg.instantiate()
```


## Prerequisites


You'll need the learning tasks; we'll assume that they lie in a directory
`../2024-gecco-tasks/2024-01-19T16-28-51.924-task-selection`.


Ensure that `http://localhost:5000` is a running Mlflow tracking server (e.g.
keep `mlflow server --no-serve-artifacts` running).


You'll probably need around 7–10 GB of RAM for some of the learning tasks.


## Hyperparameter optimization


1. From the root of this repository, run

   ```
   julia --project=. scripts/2024-iwerl/run.jl optparams
   ```
   
   to see the options of that script.
2. Run e.g.

   ```
   julia --project=. scripts/2024-iwerl/run.jl optparams 2-2-502-0-0.9-true.data.npz
   ```
   
   to optimize the hyperparameters of the configured ML algorithms for the data
   set in `2-2-502-0-0.9-true.data.npz`.
   
   Or, to optimize hyperparameters for all the task in a certain directory:
   
   ```
   julia --project=. scripts/2024-iwerl/run.jl optparams ../2024-gecco-tasks/2024-01-19T16-28-51.924-task-selection/*.npz
   ```
   
   Or, to do so parallely (make sure that 1. you have enough RAM, 2. Julia
   precompilation was already done, 3. the mlflow experiment already exists):
   
   ```
   find ../2024-gecco-tasks/2024-01-19T16-28-51.924-task-selection/*.npz -name '*.npz' -print0 | parallel -0 --results output --progress --eta julia --project=. scripts/2024-gecco/run.jl optparams '{}'
   ```


## Best-configuration runs


1. Ensure that `localhost:5000` is a running Mlflow tracking server.
2. From the root of this repository, run

   ```
   julia --project=. scripts/2024-iwerl/run.jl runbest 2-2-502-0-0.9-true.data.npz
   ```
   
   to run the configured ML algorithms using the best set of hyperparameters
   found by `optparams` for the
   `../2024-gecco-tasks/2024-01-19T16-28-51.924-task-selection/2-2-502-0-0.9-true.data.npz`
   data set.


## Slurm scripts


We provide Slurm scripts to automate performing repeated runs on a data set
(`scripts/2024-iwerl/*.sbatch`). Check `scripts/2024-gecco/submitrunbest` to get
an idea of how to automate submitting Slurm jobs for all data sets.


## Evaluation


Let's assume that we choose `foo` as the tag for this eval.


```
julia --project=. scripts/2024-iwerl/analyse.jl prep foo
```

writes a `.jls` file (containing the run data exported from mlflow) to the
current directory (you can also skip this step and use our run result data which
can be found on [Zenodo](https://doi.org/10.5281/zenodo.11143818)). That file is
then put into

```
julia --project=. scripts/2024-iwerl/analyse.jl graphs "2024 IWERL Data foo.jls" foo
```

which writes the plots to the `plots/` directory.


You can further generate/pull the convergence data from the mlflow server using

```
julia --project=. scripts/2024-iwerl/analyse.jl prepconv "2024 IWERL Data foo.jls" foo
```

which writes another `.jls` file to the current directory (or you can again skip
this and use our run result data from
[Zenodo](https://doi.org/10.5281/zenodo.11143818)).

Afterwards,

```
julia --project=. scripts/2024-iwerl/analyse.jl conv "2024 IWERL Data foo.jls" "2024 IWERL Data Fitness foo.jl" foo
```

performs and prints the results of our (rather preliminary) convergence analysis.
