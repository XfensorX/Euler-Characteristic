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
gcloud compute scp --recurse <username>@<instance name>:~/euler/results .
# !! please notice that the time of the machine is different in another region

# destroy the ressource
terraform destroy
```

Run the process in the background:

```bash
screen #start a session
# PRESS CTRL-A then CTRL-D to detach session
exit # the VM

# after login again you can see progress with
screen -r
#If it is already finished, just chekc the outputs

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
