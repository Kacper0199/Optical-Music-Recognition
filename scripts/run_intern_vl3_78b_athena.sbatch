#!/bin/bash -l
#SBATCH -J OMR_INTERNVL
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -p plgrid-gpu-a100
#SBATCH -A GRANT_NAME
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output="logs/athena_run_intern_vl3_78b.out"
#SBATCH --error="logs/athena_run_intern_vl3_78b.err"

set -e

# Project dirs
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"

if [ -f "main.py" ]; then
    PROJECT_DIR="$SUBMIT_DIR"
elif [ -f "../main.py" ]; then
    cd ..
    PROJECT_DIR=$(pwd)
else
    echo "ERROR: Cannot find main.py. Run sbatch from project root or scripts/ folder."
    exit 1
fi

echo "Project root detected at: $PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}"

export VLLM_USE_V1=0
export NCCL_DEBUG=WARN

# Scratch and cache config
export SCRATCH_DIR="/net/tscratch/people/$USER"
mkdir -p "$SCRATCH_DIR/.cache/huggingface"
mkdir -p "$SCRATCH_DIR/.cache/pip"
mkdir -p "$PROJECT_DIR/logs"

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export TMPDIR="$SCRATCH_DIR/tmp"
mkdir -p $TMPDIR

# Load modules
ml purge
module load Python/3.10.4
module load CUDA/12.4.0

echo "Loaded modules:"
module list

# .env Token HF
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Venv settings
VENV_DIR="$PROJECT_DIR/omr_venv_athena"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing requirements..."
    pip install --no-cache-dir vllm>=0.7.2 transformers>=4.50.0 accelerate>=1.3.0 flash-attn>=2.5.0
    pip install -r requirements.txt
    pip install requests
else
    echo "Activating virtual environment..."
    source $VENV_DIR/bin/activate
fi

# Huggingface login
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set in .env file!"
else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# VLLM server start
MODEL_ID="OpenGVLab/InternVL3-78B"

echo "Starting vLLM server..."

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --enforce-eager \
    --port 8000 \
    --disable-log-requests > logs/vllm_server.log 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
for i in {1..600}; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "vLLM Ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM Process died! Printing last 20 lines of logs:"
        tail -n 20 logs/vllm_server.log
        exit 1
    fi
    echo "Waiting for vLLM ($((i*5))s)..."
    sleep 5
done

# Client start
echo "Starting OMR pipeline..."
python3 main.py

# Cleaning
echo "Cleaning up..."
kill $VLLM_PID
wait $VLLM_PID
echo "Done."
