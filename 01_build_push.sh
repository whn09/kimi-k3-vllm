#!/bin/bash
# HOST-side: build the image on THIS host and push it to ECR. Run on B300-1.
#
#   bash 01_build_push.sh            # build + push $IMAGE
#   PUSH=0 bash 01_build_push.sh     # build only
#
# Then pull on the other hosts with `bash sync.sh pull-image` from the laptop.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

docker build --progress=plain -t "$IMAGE" -f Dockerfile . 2>&1 | tee "/tmp/k3v-build-${IMAGE_TAG}.log"
[[ "${PIPESTATUS[0]}" == 0 ]] || { echo "FATAL: build failed, see /tmp/k3v-build-${IMAGE_TAG}.log"; exit 1; }

[[ "${PUSH:-1}" == 1 ]] || exit 0
aws ecr describe-repositories --region "$K3V_REGION" --repository-names "$ECR_REPO" >/dev/null 2>&1 \
    || aws ecr create-repository --region "$K3V_REGION" --repository-name "$ECR_REPO" >/dev/null
aws ecr get-login-password --region "$K3V_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null
docker push "$IMAGE"
echo "pushed $IMAGE  config=$(docker image inspect -f '{{.Id}}' "$IMAGE")"
