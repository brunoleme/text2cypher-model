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

variable "attach_instance_profile" {
  description = "Attach an IAM instance profile to the EC2 instance"
  type        = bool
  default     = false
}

variable "ami_id" {
  description = "Optional explicit AMI ID to use (overrides discovery)"
  type        = string
  default     = ""
}

variable "ami_name_filter" {
  description = "AMI name wildcard to discover DLAMI PyTorch"
  type        = string
  default     = "Deep Learning AMI GPU PyTorch 2.0.* (Ubuntu 20.04) *"
}

variable "skip_ami_lookup" {
  description = "If true, skip AMI discovery (useful for destroy)"
  type        = bool
  default     = false
}

