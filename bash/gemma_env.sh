# do not run this script directly as `conda activate` doesn't work in shell scripts
conda create -n gemma3 python=3.12 -y
conda activate gemma3
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers torchao accelerate
