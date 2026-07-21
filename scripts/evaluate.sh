#!/usr/bin/env bash

# Evaluation Script

set -e

echo "Running Phase 2 Evaluation"

python training/phase2.py

echo ""

echo "Running Phase 3 Evaluation"

python training/phase3.py

echo ""
echo "Evaluation completed successfully."
