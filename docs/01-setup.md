# Setup

This document present guidelines to setup development environment for this codebase.

## Setup on Berzelius

* Load Annaconda module: `module load Anaconda/2021.05-nsc1`
* Load GCC build tool: `module load buildenv-gcccuda/11.4-8.3.1-bare`
* Create Conda environment and install packages from `environments.yml`: `conda env create -f environments.yml`
* Install MiM: `pip install mim`
* Install MMDetection: `mim install mmdet`
