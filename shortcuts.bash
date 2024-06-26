alias euler="python main.py"
alias activate_env_windows="source env/Script/activate"
alias activate_env="source env/bin/activate"
alias create_env="python -m venv env"
alias save_requirements="pip freeze > requirements.txt"
alias install_requirements="pip install -r requirements.txt"
alias download_results="gcloud compute scp --recurse philipp@phil:~/euler/results ./results"
alias start_gpu="terraform -chdir=terraform apply -var-file=gpu.tfvars -auto-approve"
alias start_cpu="terraform -chdir=terraform apply -var-file=cpu.tfvars -auto-approve"
alias stop_gpu="terraform -chdir=terraform destroy -var-file=gpu.tfvars"
alias stop_cpu="terraform -chdir=terraform destroy -var-file=cpu.tfvars"