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

# GCloud

```bash
cd terraform
terraform apply
gcloud compute ssh <instance name>

#in the VM:
sudo apt install git python3-full python3-pip
git clone https://github.com/XfensorX/Euler-Characteristic.git euler
cd euler
source shortcuts.bash
install_requirements

euler run <experiment>

# exit the VM
exit

# after experiment exit copy the results
gcloud compute scp --recurse <username>@<instance name>:~/euler/results ../results

# destroy the ressource
terraform destroy
```

# Setup

Activate shortcuts (everytime you live the terminal):

```bash
source shortcuts.bash
```

Install Libraries with:

```bash
activate_env
install_requirements
```

# Add new libraries:

Update [requirements.txt](requirements.txt) with :

```bash
save_requirements
```
