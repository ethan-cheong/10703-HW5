#!/bin/bash

source /home/ubuntu/10703-HW5/.venv/bin/activate

python muzero_code/main.py

if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
    echo "Shutting down instance in 1 minute..."
    
    sleep 60
    
    sudo shutdown -h now
else
    echo "Training failed at $(date)"
    exit 1
fi