# MLTemplate

Template Project for a Machine Learning Project using Pytorch or maybe also other ML libraries.

# Usage

```bash
euler <experiment_name>
```

for more info

```bash
euler --help
```

# Setup

Activate shortcuts (everytime you live the terminal):

```bash
source shortcuts.bash
```

Install Libraries with:

```bash
conda env create -f environment.yaml -n euler
conda activate euler
```

# Add new libraries:

Update [environment.yaml](environment.yaml) with :

```bash
conda env export | grep -v "^prefix: " > environment.yaml
```
