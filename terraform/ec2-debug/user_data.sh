#!/bin/bash
set -euxo pipefail

ENVIRONMENT="${environment}"
AWS_REGION="${aws_region}"

echo "[DEBUG EC2] Bootstrapping instance for ${ENVIRONMENT} in ${AWS_REGION}"

apt-get update -y
apt-get install -y curl unzip jq git htop

# Install AWS CLI v2
curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install || true
rm -rf /tmp/aws /tmp/awscliv2.zip

# Install Docker but do NOT start any containers
apt-get remove -y docker docker-engine docker.io || true
apt-get install -y ca-certificates gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker ubuntu || true

echo "[DEBUG EC2] Setup complete. No containers started."

