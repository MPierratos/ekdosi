# Usage

### Set your data directory

`export DATA_DIR = '<path to your data>'`, otherwise the default is set to the user cache
`~/.cache/python--glottis`

# SETUP

## Generate a conda-lock file

`./bootstrap.sh`

## Create a virtual environment with torch-gpu

```sh
conda create --name glottis --file conda-linux-64.lock
conda activate glottis
conda install lockfile html5lib 
poetry install
```

## Activate the environment 

```sh
conda activate glottis
```

## Update the environment

```sh
# Re-generate Conda lock file(s) based on environment.yml
conda-lock -k explicit --conda mamba
# Update Conda packages based on re-generated lock file
mamba update --file conda-linux-64.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```