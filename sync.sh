#!/bin/bash
# LAPTOP-side helper for the 4 B300 hosts.
#
#   bash sync.sh pull          # git pull the kit on every host (after a push)
#   bash sync.sh bundle        # no GitHub remote: ship HEAD as a git bundle and
#                              # fast-forward each host's clone from it
#   bash sync.sh pull-image    # docker pull $IMAGE on every host + compare ids
#   bash sync.sh results       # rsync hosts' results/ -> local results/
#   bash sync.sh status        # containers + GPU memory on every host
#
# The host kit dirs are git clones: push from here, `git pull` there. Never rsync
# the tree over them.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh
HOSTS="${HOSTS:-B300-1 B300-2 B300-3 B300-4}"
REMOTE="${REMOTE:-$SCRIPT_DIR_HOST}"
REPO_URL="${REPO_URL:-https://github.com/whn09/kimi-k3-vllm.git}"
SSH=(ssh -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR)

case "${1:-}" in
    pull)
        for h in $HOSTS; do
            "${SSH[@]}" "$h" "test -d $REMOTE/.git || git clone -q $REPO_URL $REMOTE; cd $REMOTE && git pull -q --ff-only && echo \"$h: \$(git log -1 --format='%h %s')\""
        done ;;
    bundle)
        B=/tmp/kimi-k3-vllm.bundle
        git bundle create -q "$B" main
        for h in $HOSTS; do
            scp -q -o LogLevel=ERROR "$B" "$h:$B"
            "${SSH[@]}" "$h" "test -d $REMOTE/.git || git clone -q -b main $B $REMOTE; cd $REMOTE && git pull -q --ff-only $B main && echo \"$h: \$(git log -1 --format='%h %s')\""
        done ;;
    pull-image)
        for h in $HOSTS; do
            ( "${SSH[@]}" "$h" "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY >/dev/null && docker pull -q $IMAGE >/dev/null && echo \"$h: \$(docker image inspect -f '{{.Id}}' $IMAGE)\"" ) &
        done; wait ;;
    results)
        mkdir -p results
        for h in $HOSTS; do rsync -az "$h:$REMOTE/results/" results/ 2>/dev/null && echo "pulled <- $h" || echo "no results on $h"; done ;;
    status)
        for h in $HOSTS; do
            echo "== $h"
            "${SSH[@]}" "$h" "docker ps --format '{{.Names}}  {{.Status}}  {{.Image}}'; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo"
        done ;;
    *) echo "usage: $0 pull|pull-image|results|status" >&2; exit 1 ;;
esac
