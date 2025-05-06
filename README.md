`scripts/generate_instructions.py` is a script for generating image editing instructions locally using Gemma 3. To run the script, make sure you have PyTorch version `<=2.6`.

`scripts/generate_descriptions.py` is a script for generating image descriptions locally using Gemma 3. To run the script, make sure you have PyTorch version `<=2.6`.

`scripts/controlnet_scribble.py` is a script for running the baseline ControlNet Scribble model.

`scripts/sd_controlnet.py` is a script for running the baseline Stable Diffusion Inpaint + ControlNet Scribble model.

These scripts should be run under different Python environments. See the `bash/` folder for how to set up the environments.

To generate instructions for the whole dataset, include the `inpainted/` and `sampled/` folders in this directory, then run `python scripts/generate_instructions.py`. The generated instructions will be in the `instructions/` folder.

To generate descriptions for the whole dataset, include the `sampled/` folder in this directory, then run `python scripts/generate_descriptions.py`. The generated descriptions will be in the `descriptions/` folder.
