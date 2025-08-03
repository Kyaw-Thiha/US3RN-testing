#!/bin/bash

set -e # Exit if any command fails

# === CONFIG ===
PROJECT_DIR="/workspace/US3RN-testing"
VENV_DIR="$PROJECT_DIR/.env"

# Usage: ./setup.sh 50629
SSH_PORT=$1
if [ -z "$SSH_PORT" ]; then
  SSH_PORT="<PORT>"
fi

REMOTE_IP=$(curl -s https://ipinfo.io/ip)

echo "📦 Installing Deadsnakes Python 3.12 and dependencies..."
apt update
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.12 python3.12-venv python3.12-dev

echo "📁 Setting up virtual environment at $VENV_DIR..."
cd "$PROJECT_DIR"
if [[ -n "$VIRTUAL_ENV" ]]; then
  deactivate
fi
rm -rf .env
python3.12 -m venv .env

echo "⬆️ Installing pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade pip

echo "📦 Installing Python requirements..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "📂 Ready. Now upload your data:"
echo "Run this on your local machine:"
echo "  scp -P $SSH_PORT -r ~/Documents/Projects/US3RN-testing/data root@$REMOTE_IP:$PROJECT_DIR/"
echo ""
echo "Run this to activate virtual environment:"
echo "  source $PROJECT_DIR/.env/bin/activate"
echo ""
echo "✅ Setup complete!"
