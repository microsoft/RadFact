# A temporary file for environment generation
TEMP_ENV_YAML := _temp_environment.yaml
DEV_ENV_YAML := dev_environment.yaml
BLACK_ARGS := --config pyproject.toml .
# Folder where all downloaded files are cached
CACHE := $(HOME)/.cache
ENV_NAME := radfact
REPO_NAME := $(shell basename `pwd`)

# Get latest Miniconda
miniconda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh

# Install the fast Conda mamba solver
mamba:
	conda update -y -n base conda
	conda install -y -n base conda-libmamba-solver
	conda config --set solver libmamba

# Create the Conda environment
env:
	@echo Removing current Conda environment
	conda env remove -n $(ENV_NAME)
	@echo Re-building Conda environment from $(DEV_ENV_YAML)
	conda env create -n $(ENV_NAME) -f $(DEV_ENV_YAML)
	@echo Add all local packages in editable mode
	$(MAKE) setup_packages_with_deps

# Remove and re-create the current environment, based on primary_deps.yaml.
# Then add all packages in editable mode, without installing the dependencies.
env_from_primary:
	@echo Removing current Conda environment
	conda env remove -n $(ENV_NAME)
	@echo Re-building Conda environment from primary_deps.yaml
	conda env create -f primary_deps.yaml -n $(ENV_NAME)

# Run the command that adds all subpackages in editable mode, without adding the dependencies.
# Dependencies are already installed in the Conda environment.
setup_packages:
	conda run -n $(ENV_NAME) --no-capture-output python setup_packages.py --no-deps

# Run the command that adds all subpackages in editable mode, including the dependencies. This should be used when
# creating the locked environment file.
setup_packages_with_deps:
	conda run -n $(ENV_NAME) --no-capture-output python setup_packages.py --add-dev-deps

# Export the Conda environment to _temp_environment.yaml
# Exclude all the local packages that are possibly installed.
_env_export:
	rm -f $(TEMP_ENV_YAML)
	@echo "# DO NOT MANUALLY EDIT THIS FILE, USE 'make env_lock' INSTEAD" > $(TEMP_ENV_YAML)
	conda env export --no-builds -n $(ENV_NAME) | \
		grep -v "^prefix" | \
		grep -Ev "\s(azure-utils|llm-utils|radfact)==" \
		>> $(TEMP_ENV_YAML); \

# Export the Conda environment to dev_environment.yaml
# We need to call Make recursively here for _env_export because in environment locking this will be called twice
# and would be skipped if specified as 'env_export: _env_export'
env_export:
	$(MAKE) _env_export
	mv $(TEMP_ENV_YAML) $(DEV_ENV_YAML)

# This make target will re-build the Conda environment based on primary_deps.yaml, install all packages,
# export the locked environment
env_lock_in_container: env_from_primary setup_packages_with_deps env_export

# Full Docker-based solution to re-create the environments:
env_lock:
	@echo If something goes wrong here, you may need to execute 'docker rm -f env-container' afterwards to clean up
	@echo Building the base docker image - this will take several minutes
	docker build -t devcont -f Dockerfile_no_conda .
	@echo Kill any container that may still be running from previous failed attempts
	docker rm -f env-container || true
	@echo Start a container with that image, with a sleep command to prevent it from terminating immediately
	docker run -d --name env-container devcont sleep infinity
	@echo Copy the repository to the container
	docker cp `pwd` env-container:/home/radfact
	@echo Change the copied files to belong to the "radfact" user that all commands are running under
	docker exec -u root env-container chown -R radfact:radfact /home/radfact/$(REPO_NAME)
	@echo Starting environment re-build - this will take several minutes
	docker exec -it env-container bash -c "cd $(REPO_NAME) && make env_lock_in_container"
	@echo Copying updated environment files out of the container
	docker cp env-container:/home/radfact/$(REPO_NAME)/$(DEV_ENV_YAML) .
	@echo Stopping container
	docker rm -f env-container


lfs:
	@if [ -z "$$(which git-lfs)" ]; then \
		echo "Installing git-lfs"; \
		sudo apt-get install git-lfs ; \
	else \
		echo "git-lfs already installed" ; \
	fi
	git lfs install
	git lfs pull

# Run black and reformat all files as necessary
black:
	black $(BLACK_ARGS)

# Run black, but do not reformat files
blackcheck:
	black --check $(BLACK_ARGS)

# Run flake8
flake8:
	flake8 --config .flake8 .

# Run mypy
mypy:
	mypy .

# Run black to reformat all files, then flake8 to find issues beyond formatting, then mypy
check: black flake8 mypy
