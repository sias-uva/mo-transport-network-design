# mo-transport-network-design
Designing Transportation Networks with Multiple Objectives


# Setup
Create the conda environment:
```
conda env create -f environment.yml
```

Install the mo-tndp environment:
```
git submodule update --init --recursive
cd envs/mo-tndp
pip install -e .
```

Install the morl-baselines repo:
```
cd morl-baselines
pip install -e .
```

On Mac M1, if you get an error while installing the morl_baselines, do `pip install osqp==0.6.1` and try again ([source](https://stackoverflow.com/questions/65920955/failed-building-wheel-for-qdldl-when-installing-cvxpy))
