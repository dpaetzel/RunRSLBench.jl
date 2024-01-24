# Scripts to perform the experiments for the 2024 GECCO paper


All of these commands are assumed to be run from within a Nix shell environment
which can be entered by doing at the root of the repository

```
nix develop --impure
```


Install Julia depenencies by opening a Julia shell (`julia --project=.`) and then doing

```
import Pkg; Pkg.instantiate()
```


## Hyperparameter optimization


1. Ensure that `localhost:5000` is a running Mlflow tracking server (e.g.
   `mlflow server --no-serve-artifacts`).
2. From the root of this repository, run

   ```
   julia --project=. scripts/2024-gecco/run.jl optparams
   ```
   
   to see the options of that script.
3. Run e.g.

   ```
   julia --project=. scripts/2024-gecco/run.jl optparams 2-2-502-0-0.9-true.data.npz
   ```
   
   to optimize the hyperparameters of the configured ML algorithms for the data
   set in `2-2-502-0-0.9-true.data.npz`.


## Best-configuration runs


1. Ensure that `localhost:5000` is a running Mlflow tracking server.
2. From the root of this repository, run

   ```
   julia --project=. scripts/2024-gecco/run.jl runbest 2-2-502-0-0.9-true.data.npz
   ```
   
   to run the configured ML algorithms using the best set of hyperparameters
   found by `optparams`.

