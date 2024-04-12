provider "google" {
  project = "idyllic-chimera-419902"
  region  = "us-west1"
}

variable "CPUs" {
  type    = number
  default = 2
}

variable "NAME" {
  type        = string
  description = "The name of the VM"
}


variable "MEMORY" {
  type    = number
  default = 4096
}

variable "USE_CUDA" {
  type    = bool
  default = false

}

data "google_compute_image" "used_image" {
  family  = "pytorch-latest-cpu"
  project = "deeplearning-platform-release"
}


resource "google_compute_instance" "vm_instance" {

  name         = var.NAME
  description  = "Used for computing euler characteristics"
  machine_type = "custom-${var.CPUs}-${var.MEMORY}"
  zone         = "us-west1-a"

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
}
