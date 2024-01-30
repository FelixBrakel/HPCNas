---
tags:
  - "#flexflow"
links: 
type: documentation
---
# Overview
The goal of this guide is to get an installation of flexflow with the following goals:
 - Without using root privileges

# 0.1 Installing conda
On the DAS you have to install conda yourself. I recommend installing it to the /var/scratch/user directory as it will exceed the disk quota of the home directory quite quickly.

Download the miniconda install script and execute it:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```

And then depending on your shell:
```
~/miniconda3/bin/conda init bash
```
Or
```
~/miniconda3/bin/conda init zsh
```

# 1 Clone the repository
Clone the repository to your scratch folder and move into the folder:
```
git clone --recursive https://github.com/flexflow/FlexFlow.git
cd FlexFlow
```
Although at the time of writing the latest commit on the inference branch will compile I recommend switching to the v23.11.0 release. There is also the v23.12.0 release but I was unable to get this version to compile.
```
git checkout tags/v23.11.0
```
# 2 Create the conda environment
In the FlexFlow root directory execute the following commands to create a conda environment, install the dependencies and activate it:
```
conda env create -f conda/environment.yml
conda activate hpcnas
```
The conda environment should be active for the entire compilation/installation procedure.

Then install these additional packages
```
conda install -c conda-forge libgcc-ng=12 libstdcxx-ng=12 conda-build python=3.7
```
Install the python requirements
```
pip install -r requirements.txt
```
# 3 Configure and build
Create a new folder `build` and configure the build in there:
```
mkdir build
cd build
CC=gcc-12 CXX=g++-12 CUDA_DIR=/opt/cuda FF_BUILD_ALL_EXAMPLES=ON FF_USE_PYTHON=ON ../config/config.linux
```

Then start the build and grab a coffee, even on the DAS this will take a while.
```
make -j16
```

# 4 Install into the conda environment
Installing FlexFlow into the conda env is a bit hacky but quite useful allows using FlexFlow as a python package in the conda environment as well as a C++ library without the need for root privileges as would be when installing it as a system package.

First create a new folder in the FlexFlow root directory and create two new files `build.sh` and `meta.yaml`
```
export FF_HOME=/var/scratch/$USER/FlexFlow
mkdir conda-recipe
cd conda-recipe
touch build.sh
touch meta.yaml
```

Then edit the two files such that they have the following content:

build.sh
```sh
#! /bin/bash  
  
cd ${FF_HOME}/build  
make install
```

meta.yaml:
```
package:  
 name: flexflow  
 version: 0.1  
build:  
 script_env:  
   - FF_HOME
```

Now execute the following command to build the conda package:
```
conda build .
```

Then install it using the following command

```
conda install --offline $CONDA_PREFIX/conda-bld/linux-64/flexflow-0.1-0.tar.bz2
```

# 5 Verify installation 
To verify the installation was successful we run an example. This example should be present in the PATH as we compiled with FF_BUILD_ALL_EXAMPLES=ON:
```
alexnet -ll:gpu 1 -ll:fsize 8000 -ll:zsize 8000
```

# 6 Install NASLib
```
git clone https://github.com/automl/NASLib.git
cd NASLib
```

```
conda install pytorch torchvision torchaudio -c pytorch
```

```
pip install tornado
pip install -e .
```