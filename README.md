# distributed-aerial-transportation

### Requirements

- Core packages:
  - Collision detection: `coal` (previously `hppfcl`)
  - Dynamics: `pinocchio`
  - Optimization: <br>
    Modelling language: `cvxpy` <br>
    Conic solver: `clarabel`

- Visualization packages:
  - Plotting: `matplotlib`
  - 3D: `meshcat-dev` should be installed from [source](https://github.com/meshcat-dev/meshcat-python).

- Formatting packages (optional):
  - Code formatting: `black`
  - Import ordering: `isort`

- Other packages:
  - Progress meter: `tqdm`

---

### Setup

    export PYTHONPATH=$PYTHONPATH:pwd
    export PYTHONOPTIMIZE=TRUE
