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


Ensure that `http://localhost:5000` is a running Mlflow tracking server (e.g.
keep `mlflow server --no-serve-artifacts` running).


You'll probably need around 7–10 GB of RAM for some of the learning tasks.


# Performing runs


1. Ensure that `localhost:5000` is a running Mlflow tracking server.
2. From the root of this repository, run

   ```
   julia --project=. scripts/2024-gecco/run.jl runbest 2-2-502-0-0.9-true.data.npz
   ```
   
   to run the configured ML algorithms on the
   `../2024-gecco-tasks/2024-01-19T16-28-51.924-task-selection/2-2-502-0-0.9-true.data.npz`
   data set.
   
   
## Slurm scripts


We provide a Slurm script to automate performing repeated runs on all the data
sets (`submitrunbest`).


## Evaluation


You can choose a tag which is appended to the files (so already existing files
are not overwritten). Let's assume you choose `mytag`.

```
julia --project=. scripts/2024-gecco/analyse.jl prep mytag
```

writes a `.jls` file (containing the run data exported from mlflow) to the
current directory. You can also skip this step and use our run result data which
can be found at [TODO]. That `.jls` file is then put into

```
julia --project=. scripts/2024-gecco/analyse.jl graphs "2024 GECCO Data mytag.jls" mytag
```

which writes the plots to the `plots/` directory.
