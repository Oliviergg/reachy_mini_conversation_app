#!/usr/bin/env bash
# Deploy the conversation app to a Reachy Mini Pi (wireless install).
#
# Prerequisites:
#   - SSH access to the robot (default: pollen@10.0.0.44)
#   - sshpass (Mac: `brew install sshpass`)
#   - rsync (preinstalled on macOS)
#
# Usage:
#   REACHY_SSH_PASS=root scripts/deploy_to_robot.sh
#   # or
#   ROBOT_HOST=10.0.0.44 ROBOT_USER=pollen REACHY_SSH_PASS=root scripts/deploy_to_robot.sh

set -euo pipefail

ROBOT_HOST="${ROBOT_HOST:-10.0.0.44}"
ROBOT_USER="${ROBOT_USER:-pollen}"
REMOTE_DIR="${REMOTE_DIR:-/home/${ROBOT_USER}/reachy_mini_conversation_app}"
SSH_PASS="${REACHY_SSH_PASS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${SSH_PASS}" ]]; then
    echo "ERROR: set REACHY_SSH_PASS=<password>" >&2
    exit 1
fi

if ! command -v sshpass >/dev/null; then
    echo "ERROR: sshpass missing. Install with: brew install sshpass" >&2
    exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR)
SSH_TARGET="${ROBOT_USER}@${ROBOT_HOST}"
RSYNC_SSH="sshpass -p ${SSH_PASS} ssh ${SSH_OPTS[*]}"
SSH_RUN=(sshpass -p "${SSH_PASS}" ssh "${SSH_OPTS[@]}" "${SSH_TARGET}")

echo "=== Probing Pi (${SSH_TARGET}) ==="
"${SSH_RUN[@]}" 'uname -m; python3 --version; ls -d ~/apps_venv 2>/dev/null || echo "(no apps_venv yet)"'

echo
echo "=== Syncing source to ${REMOTE_DIR} ==="
sshpass -p "${SSH_PASS}" rsync -az --delete \
    -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.git' --exclude 'cache' --exclude 'dist' --exclude 'build' \
    --exclude '*.egg-info' --exclude 'profiles/user_personalities' \
    --exclude '.env' \
    "${REPO_ROOT}/" "${SSH_TARGET}:${REMOTE_DIR}/"

echo
echo "=== Installing into apps_venv ==="
"${SSH_RUN[@]}" bash -s <<EOSH
set -euo pipefail
APPS_VENV="\$HOME/apps_venv"
if [[ ! -d "\$APPS_VENV" ]]; then
    echo "Creating \$APPS_VENV ..."
    python3 -m venv "\$APPS_VENV"
    "\$APPS_VENV/bin/pip" install --upgrade pip
fi
"\$APPS_VENV/bin/pip" install -U "${REMOTE_DIR}"
# openwakeword pulls tflite-runtime as a hard dep on Linux but we use onnxruntime;
# tflite-runtime has no wheels for recent Python on aarch64, so install with --no-deps.
"\$APPS_VENV/bin/pip" install -U --no-deps "openwakeword>=0.6.0"
"\$APPS_VENV/bin/pip" install -U "onnxruntime>=1.17" "scikit-learn" "tqdm" "requests"
echo "Installed. Entry point:"
"\$APPS_VENV/bin/reachy-mini-conversation-app" --help >/dev/null && echo "  CLI OK"
EOSH

echo
echo "=== .env ==="
if "${SSH_RUN[@]}" "test -f ${REMOTE_DIR}/.env"; then
    echo "  remote .env already present, leaving untouched."
else
    echo "  no remote .env found. Pushing a template."
    sshpass -p "${SSH_PASS}" rsync -az -e "ssh ${SSH_OPTS[*]}" \
        "${REPO_ROOT}/src/reachy_mini_conversation_app/.env.example" \
        "${SSH_TARGET}:${REMOTE_DIR}/.env" 2>/dev/null \
        || echo "  (no .env.example to push; create ${REMOTE_DIR}/.env on the robot manually)"
fi

cat <<EOM

=== Done ===
On the robot, the app entry point is:
  ~/apps_venv/bin/reachy-mini-conversation-app

The desktop "Reachy Mini Control" should now list it under Installed apps
(it scans the apps_venv automatically).

Don't forget to set on the robot:
  ${REMOTE_DIR}/.env  ->  OPENAI_API_KEY=...
                          REACHY_WAKE_WORD=hey_jarvis
                          REACHY_WAKE_WORD_SLEEP_TIMEOUT_S=30
EOM
