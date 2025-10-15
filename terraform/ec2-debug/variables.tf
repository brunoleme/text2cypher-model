variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g4dn.xlarge"
}

variable "ssh_key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "use_existing_instance_profile" {
  description = "If true, use an existing IAM instance profile instead of creating one"
  type        = bool
  default     = false
}

variable "existing_instance_profile_name" {
  description = "Existing IAM instance profile name to attach to EC2 when use_existing_instance_profile=true"
  type        = string
  default     = ""
}

