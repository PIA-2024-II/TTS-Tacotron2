#!/bin/bash

# Clone Tacotron 2 repository
git clone https://github.com/NVIDIA/tacotron2.git
cd tacotron2

# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Check CUDA and PyTorch compatibility
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__version__)"

# Test the base model with examples included in the repo
python inference.py --checkpoint_file=pretrained_model.pt --input_text="Hello, world!"