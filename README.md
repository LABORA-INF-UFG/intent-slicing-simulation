# rrm-slice-rl

Code containing RRM simulation using RL in a scenario with RAN slicing.

## Install

- Install packages needed for Pipenv:
```bash
sudo apt update
sudo apt install \
    build-essential \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev
```
- Install [pyenv](https://github.com/pyenv/pyenv)
- Install [pipenv](https://github.com/pypa/pipenv) with `pip install pipenv`
- Make pipenv create a `.venv` folder inside the project directory setting the environmental variable `PIPENV_VENV_IN_PROJECT` with `export PIPENV_VENV_IN_PROJECT=1`
- Install dependencies using pipenv: `pipenv install`
- To access the virtual environment created, run `pipenv shell`, now all commands which you run will be performed into virtual enviroment created
- (In case you want to contribute with this repo, if not you can skip this step) Activate pre-commit hooks to use [black formatter](https://github.com/psf/black), [flake8 lint](https://gitlab.com/pycqa/flake8) and [Isort references](https://github.com/timothycrosley/isort). Run `pre-commit install`. Now every time you make a commit, black formatter, flake8 and isort will make tests to verify if your code is following the [patterns](https://realpython.com/python-pep8/) (you can adapt your IDE or text editor to follow this patterns, e.g. [vs code](https://code.visualstudio.com/docs/python/python-tutorial#_next-steps))

## Hyperparameters optimization using Optuna

Run the script `optimize.py` using pipenv running `pipenv run python optimize.py`. It would take a long time to Optuna generate the optimized hyperparameters for each scenario, so you can use the hyperparameters that it were already generated into `hyperparameter_opt` folder and **skip this step**.
## Training and testing ML model 

Run the command `pipenv python run.py` to start the simulation for all scenarios. You can watch the training performance using tensorboard. After the training finish, the RL models will be saved into `agents` folder, the VecNormalize parameters into `vecnormalize_models` folder, the evaluations made along with the trainign into `evaluations` folder, and the test results into `hist` folder.
## Generating figures with the results

To generate the figures with results obtained in the paper, you can run `pipenv run python plot_results.py` and the figures should be available into the folder `results" as soon as the script finish.
