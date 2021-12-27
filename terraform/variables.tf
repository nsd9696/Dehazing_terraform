variable "project_id" {
  description = "project id"
  default = "terraform-test-336308"
}

variable "region" {
  description = "region"
  default = "asia-northeast2"
}

variable "storage_class"{
  description = "bucket-class"
  default = "standard"
}

variable "gke_num_nodes" {
  default     = 2
  description = "number of gke nodes"
}

variable "gke_machine_type" {
  default     = "e2-medium"
  description = "machine type of gke nodes"
}

variable "github_owner" {
    default = "nsd9696"
}

variable "github_repository" {
    default = "terraform-test"
}

variable "branch_name" {
    default = "master"
}