#!/bin/bash
# Local development environment runner for Entity Resolution Pipeline
# This script sets the environment to 'local' and runs the pipeline with development settings

export PIPELINE_ENV=local

echo "Starting Entity Resolution Pipeline in LOCAL development mode"
echo "Using moderate resource settings for development"
echo "Environment: $PIPELINE_ENV"

# Run the pipeline with any arguments passed to this script
python -m src.orchestrating "$@"