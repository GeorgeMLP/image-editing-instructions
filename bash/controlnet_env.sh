# do not run this script directly as `conda activate` doesn't work in shell scripts
conda create -n controlnet python=3.12 -y
conda activate controlnet
pip install controlnet_aux
pip install diffusers transformers accelerate
pip install xformers
pip install mediapipe
