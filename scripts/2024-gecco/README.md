# Scripts to perform the experiments for the 2024 GECCO paper


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


Since the `RSLModels.jl` package has not been published yet, you'll need a local
copy of that package. We'll assume it's located at `../RSLModels.jl` (this is
the path used in this `RunRSLBench.jl` project; use `]dev` etc. in a Julia
project REPL to change that path).


Ensure that `http://localhost:5000` is a running Mlflow tracking server (e.g.
keep `mlflow server --no-serve-artifacts` running).


You'll probably need around 7–10 GB of RAM for some of the learning tasks.


## Hyperparameter optimization


1. From the root of this repository, run

   ```
   julia --project=. scripts/2024-gecco/run.jl optparams
   ```
   
   to see the options of that script.
2. Run e.g.

   ```
   julia --project=. scripts/2024-gecco/run.jl optparams 2-2-502-0-0.9-true.data.npz
   ```
   
   to optimize the hyperparameters of the configured ML algorithms for the data
   set in `2-2-502-0-0.9-true.data.npz`.
   
   Or, to optimize hyperparameters for all the task in a certain directory:
   
   ```
   julia --project=. scripts/2024-gecco/run.jl optparams ~/2024-gecco-tasks/2024-01-19T16-28-51.924-task-selection/*.npz
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
   julia --project=. scripts/2024-gecco/run.jl runbest 2-2-502-0-0.9-true.data.npz
   ```
   
   to run the configured ML algorithms using the best set of hyperparameters
   found by `optparams`.


## Evaluation


```
julia --project=. scripts/2024-gecco/analyse.jl prep
```

writes a `.jls` file (containing the run data exported from mlflow) to the
current directory (you can also skip this step and use our run result data which
can be found at [TODO]). That file is then put into

```
julia --project=. scripts/2024-gecco/analyse.jl graphs "2024 GECCO Data.jl"
```

which writes the plots to the `plots/` directory.


You can further generate/pull the convergence data from the mlflow server using

```
julia --project=. scripts/2024-gecco/analyse.jl prepconv "2024 GECCO Data.jl"
```

which writes another `.jls` file to the current directory (or you can again skip
this and use our run result data from [TODO]).

Afterwards,

```
julia --project=. scripts/2024-gecco/analyse.jl prepconv "2024 GECCO Data.jl" "2024 GECCO Data Fitnesses.jl"
```

performs and prints the results of our (rather preliminary) convergence analysis.
