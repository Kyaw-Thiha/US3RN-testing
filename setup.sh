#!/bin/bash

set -e # Exit if any command fails

# === CONFIG ===
PYTHON_VERSION="3.12.3"
INSTALL_DIR="/usr/local"
PROJECT_DIR="/workspace/US3RN-testing"
VENV_DIR="$PROJECT_DIR/.env"

echo "📦 Installing dependencies..."
apt update && apt install -y \
  build-essential zlib1g-dev libssl-dev \
  libncurses-dev libffi-dev libsqlite3-dev \
  libbz2-dev wget curl

echo "🐍 Installing Python $PYTHON_VERSION..."
cd /tmp
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -xf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION
./configure --enable-optimizations
make -j$(nproc)
make altinstall # installs as python3.12 without overwriting system python

echo "📁 Setting up project at $PROJECT_DIR..."
git clone https://github.com/Kyaw-Thiha/US3RN-testing.git
cd "$PROJECT_DIR"

/usr/local/bin/python3.12 -m venv .env
source .env/bin/activate

echo "⬆️ Upgrading pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade pip

echo "Then activate the virtual environment and run:"
echo "  source $VENV_DIR/bin/activate && pip install -r $PROJECT_DIR/requirements.txt"

echo "📂 Ready. Now upload your data:"
echo ""
echo "Run this on your local machine:"
echo "  scp -P 50629 -r ~/Documents/Projects/US3RN-testing/data root@<REMOTE_IP>:$PROJECT_DIR/"
echo ""

echo "✅ Setup complete!"
