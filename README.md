# Euler Characteristic

Trying to predict Euler Characteristic with different Neural Network Based Models.

# Usage

```bash
euler run <experiment_name>
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
