# mo-transport-network-design
We introduce Lorenz Conditioned Networks (LCN), a novel multi-policy algorithm for addressing fairness in Multi-Objective Reinforcement Learning (MORL). Based on Lorenz optimality, LCN learns policies that ensure a fair distribution of rewards among different objectives. We extend LCN to introduce $\lambda$-LCN, based on a relaxation of Lorenz optimality that offers flexibility in determining fairness preferences. Finally, we address the lack of real-world MORL benchmarks, by introducing a large-scale, multi-objective environment for real-world transportation network design. Experiments in Xi'an and Amsterdam demonstrate LCN's ability to learn fair policies and scalability in high-dimensional state-action and reward spaces.


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

Install the [deep-sea-treasure](https://github.com/imec-idlab/deep-sea-treasure) environment:
```
python3 -m pip install  deep_sea_treasure
```

To run GPI-PD: Install [pycddlib](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation)
First install gmp:
```
brew install gmp
```
Then, to install the package using pip:
```
CFLAGS=-I`brew --prefix gmp`/include LDFLAGS=-L`brew --prefix gmp`/lib pip install pip install pycddlib
```

On Mac M1, if you get an error while installing the morl_baselines, do `pip install osqp==0.6.1` and try again ([source](https://stackoverflow.com/questions/65920955/failed-building-wheel-for-qdldl-when-installing-cvxpy))

On a linux cluster without sudo permissions, use the following istructions.
Firstly, cd to your home directory.
```
wget https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.bz2
tar -xjf gmp-6.3.0.tar.bz2
cd gmp-6.3.0/
./configure --prefix=/home/YOUR_USER_NAME/opt/
```
Then, configure the lib and include paths:
```
export LD_LIBRARY_PATH=/home/YOUR_USER_NAME/opt/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/YOUR_USER_NAME/opt/include:$C_INCLUDE_PATH
```

And finally, install pycddlib:
```
CFLAGS=-I/home/dmichai/opt/include LDFLAGS=-L/home/dmichai/opt/lib pip install pycddlib
```

# Reproducing the Experiments
All commands to reproduce the experiments can be found [here](https://aware-night-ab1.notion.site/Project-B-MO-LCN-Experiment-Tracker-b4d21ab160eb458a9cff9ab9314606a7)
