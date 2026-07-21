#!/usr/bin/env bash

# Training Script
# Vision-Language Image-Text Alignment Framework

set -e

echo "Starting Phase 1 Training"

python training/phase1.py

echo ""
echo "Training completed successfully."
