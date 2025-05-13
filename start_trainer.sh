#!/bin/bash
# Start the model trainer with proper configuration

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Check if Redis is running, start if not
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis server..."
    "$SCRIPT_DIR/redis_integration/start_redis.sh" &
    sleep 2  # Give Redis time to start
fi

# Start the trainer
echo "Starting model trainer..."
python3 "$SCRIPT_DIR/src/ml/trainer/train.py" > "$LOG_DIR/trainer.log" 2>&1 &
TRAINER_PID=$!
echo $TRAINER_PID > "$SCRIPT_DIR/.trainer.pid"
echo "Model trainer started with PID: $TRAINER_PID"
