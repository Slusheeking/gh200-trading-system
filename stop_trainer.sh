#!/bin/bash
# Stop the model trainer gracefully

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if PID file exists
if [ -f "$SCRIPT_DIR/.trainer.pid" ]; then
    TRAINER_PID=$(cat "$SCRIPT_DIR/.trainer.pid")
    echo "Stopping model trainer (PID: $TRAINER_PID)..."
    kill -TERM $TRAINER_PID 2>/dev/null || true
    sleep 5
    # Force kill if still running
    if kill -0 $TRAINER_PID 2>/dev/null; then
        echo "Model trainer did not stop gracefully, forcing..."
        kill -9 $TRAINER_PID 2>/dev/null || true
    fi
    rm -f "$SCRIPT_DIR/.trainer.pid"
else
    # Try to find and kill by command
    echo "Stopping model trainer..."
    pkill -f "python3 $SCRIPT_DIR/src/ml/trainer/train.py" || true
fi

echo "Model trainer stopped"
