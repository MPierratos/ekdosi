# Create a bootstrap env
conda create -p /tmp/bootstrap -c conda-forge mamba conda-lock poetry='1.*'
conda activate /tmp/bootstrap

# Create Conda lock file(s) from environment.yml
conda-lock -k explicit --conda mamba
# Set up Poetry
poetry init --python=~3.11  # version spec should match the one from environment.yml
# Fix package versions installed by Conda to prevent upgrades
poetry add --lock torch=2.1.0 torchaudio=2.1.0 torchvision=0.16.0
# Add conda-lock (and other packages, as needed) to pyproject.toml and poetry.lock
poetry add --lock conda-lock

# Remove the bootstrap env
conda deactivate
rm -rf /tmp/bootstrap

# Add Conda spec and lock files
# git add environment.yml virtual-packages.yml conda-linux-64.lock
# # Add Poetry spec and lock files
# git add pyproject.toml poetry.lock
# git commit