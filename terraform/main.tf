#TODO: Refactor


variable "REGION" {
  type    = string
  default = "us-west1"
}

variable "ZONE" {
  type    = string
  default = "a"
}

provider "google" {
  project = "idyllic-chimera-419902"
  region  = "us-west1"
}

variable "NAME" {
  type        = string
  description = "The name of the VM"
}

variable "GPU_TYPE" {
  type    = string
  default = "nvidia-test-t4"
}
variable "GPU_COUNT" {
  type    = number
  default = 1
}

variable "CPUs" {
  type    = number
  default = 2
}

variable "MEMORY" {
  type    = number
  default = 4096
}

variable "USE_GPU" {
  type = bool
}

data "google_compute_image" "used_image" {
  family  = var.USE_GPU ? "pytorch-latest-gpu" : "pytorch-latest-cpu"
  project = "deeplearning-platform-release"
}


resource "google_compute_instance" "vm_instance" {

  name         = var.NAME
  description  = "Used for computing euler characteristics"
  machine_type = var.USE_GPU ? "n1-standard-1" : "custom-${var.CPUs}-${var.MEMORY}"
  zone         = "${var.REGION}-${var.ZONE}"

  boot_disk {
    initialize_params {
      #TODO: Add GPU Support
      image = data.google_compute_image.used_image.self_link
    }
  }
  network_interface {
    network = "default"
    access_config {}
  }
  allow_stopping_for_update = true
  metadata = {
    "install-nvidia-driver" = var.USE_GPU ? true : false
  }

  guest_accelerator = var.USE_GPU ? [{
    type  = var.GPU_TYPE
    count = var.GPU_COUNT
  }] : null

  scheduling {
    on_host_maintenance = "TERMINATE"
  }


}
