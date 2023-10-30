# Usage

### Set your data directory

`export DATA_DIR = '<path to your data>'`, otherwise the default is set to the user cache
`~/.cache/python--ekdosi`

# SETUP

## Create a virtual environment with torch-gpu

```sh
conda create --name ekdosi --file conda-linux-64.lock
conda activate ekdosi
conda install lockfile html5lib 
poetry install
```

## Activate the environment 

```sh
conda activate ekdosi
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