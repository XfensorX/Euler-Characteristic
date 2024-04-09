# MLTemplate

Template Project for a Machine Learning Project using Pytorch or maybe also other ML libraries.

# Usage

```bash
python main.py run <experiment_name>
```

for more info

```bash
python main.py --help
```

# Setup

Install Libraries with:

```bash
conda env create -f environment.yaml
```

# Add new libraries:

Update [environment.yaml](environment.yaml) with :

```bash
conda env export | grep -v "^prefix: " > environment.yaml
```
