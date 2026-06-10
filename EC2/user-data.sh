#!/bin/bash
# Bootstrap EC2 for Streamlit Credit Score App

set -eu

# ---------------- EDIT ----------------
GIT_REPO="https://github.com/<username>/<repo>.git"
SUBFOLDER="EC2"                  
APP_FILE="streamlit_app.py"
# --------------------------------------

REGION="us-east-1"

APP_DIR="/opt/credit-score-app"
VENV_DIR="/opt/streamlit-venv"

if [ -z "$SUBFOLDER" ]; then
    APP_PATH="$APP_DIR"
else
    APP_PATH="$APP_DIR/$SUBFOLDER"
fi

# Update system
dnf update -y

# Install packages
dnf install -y \
    python3 \
    python3-pip \
    git

# Clone repository
git clone "$GIT_REPO" "$APP_DIR"

chown -R ec2-user:ec2-user "$APP_DIR"

# Check app exists
if [ ! -f "$APP_PATH/$APP_FILE" ]; then
    echo "FATAL: $APP_PATH/$APP_FILE not found"

    find "$APP_DIR" -maxdepth 4 -type f

    exit 1
fi

# Create virtual environment
python3 -m venv "$VENV_DIR"

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

# Create systemd service
cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Credit Score Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH

ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable streamlit
systemctl start streamlit

sleep 5

if systemctl is-active --quiet streamlit; then
    echo "================================="
    echo "Streamlit started successfully."
    echo "================================="

    touch "$APP_DIR/.userdata-success"
    chown ec2-user:ec2-user "$APP_DIR/.userdata-success"
else
    echo "================================="
    echo "Streamlit failed to start."
    echo "================================="

    journalctl -u streamlit -n 50 --no-pager || true

    exit 1
fi