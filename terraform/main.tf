
terraform {
  required_version = ">= 0.12"
}
provider "google" {
  version = "~> 3.46"
  project = var.project_id
  region  = var.region
}

# VPC
resource "google_compute_network" "vpc" {
  name                    = "${var.project_id}-vpc"
  auto_create_subnetworks = "false"
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.project_id}-subnet"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.10.0.0/24"
}

# Storage bucket
resource "google_storage_bucket" "bucket" {
  project = var.project_id
  name = "${var.project_id}-model-bucket"
  location = var.region
  force_destroy = true
  storage_class = var.storage_class
  versioning {
    enabled = true
  }
}

# Access bucket
resource "google_storage_bucket_access_control" "public_rule" {
  bucket = google_storage_bucket.bucket.name
  role   = "READER"
  entity = "allUsers"
}

#Firewall
resource "google_compute_firewall" "default" {
  name    = "test-firewall"
  network = google_compute_network.vpc.name

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
    ports    = ["80", "5601", "8501"]
  }

  source_tags = ["web"]
}

# GKE cluster
resource "google_container_cluster" "primary" {
  name     = "${var.project_id}-gke"
  location = var.region

  min_master_version = "1.19"

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
}

# Separately Managed Node Pool
resource "google_container_node_pool" "primary_nodes" {
  name       = "${google_container_cluster.primary.name}-node-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.gke_num_nodes

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]

    labels = {
      env = var.project_id
    }

    machine_type = var.gke_machine_type
    tags         = ["gke-node", "${var.project_id}-gke"]

    metadata = {
      # From GKE 1.12 onwards, disable-legacy-endpoints is set to true by the API.
      # If metadata is set but that default value is not included, Terraform will attempt to unset the value. 
      # To avoid this, set the value in your config.
      disable-legacy-endpoints = "true"
    }
  }
}

resource "google_cloudbuild_trigger" "cloud_build_trigger" {
  provider    = "google"
  description = "GitHub Repository Trigger ${var.github_owner}/${var.github_repository} (${var.branch_name})"

  github {
    owner = var.github_owner
    name  = var.github_repository
    push {
      branch = var.branch_name
    }
  }

#   substitutions = {
#     _GCR_REGION           = var.gcr_region
#     _GKE_CLUSTER_LOCATION = var.location
#     _GKE_CLUSTER_NAME     = var.cluster_name
#   }

  # The filename argument instructs Cloud Build to look for a file in the root of the repository.
  # Either a filename or build template (below) must be provided.
#   autodetect = True
  filename = "cloudbuild.yaml"
}