#!/bin/bash
# Production environment runner for Entity Resolution Pipeline
# This script sets the environment to 'prod' and runs the pipeline with high-performance settings

export PIPELINE_ENV=prod

echo "Starting Entity Resolution Pipeline in PRODUCTION mode"
echo "Using 64 cores and 247GB RAM optimized settings"
echo "Environment: $PIPELINE_ENV"

# Run the pipeline with any arguments passed to this script
python -m src.orchestrating "$@"