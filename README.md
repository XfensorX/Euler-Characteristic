# MLTemplate

Template Project for a Machine Learning Project using Pytorch or maybe also other ML libraries.

# Setup

Install Libraries with:

```bash
conda env create -f environment.yml
```

# Add new libraries:

Update [environment.yml](environment.yml) with :

```bash
conda env export | grep -v "^prefix: " > environment.yml
```
