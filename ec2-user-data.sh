#!/bin/bash
set -euxo pipefail

dnf install -y docker
systemctl enable --now docker

aws ecr get-login-password --region ap-south-1 |
  docker login --username AWS --password-stdin 660743084708.dkr.ecr.ap-south-1.amazonaws.com

docker pull 660743084708.dkr.ecr.ap-south-1.amazonaws.com/sugarcane-web:latest

docker run -d \
  --name sugarcane-api \
  --restart unless-stopped \
  -p 8000:8000 \
  660743084708.dkr.ecr.ap-south-1.amazonaws.com/sugarcane-web:latest
