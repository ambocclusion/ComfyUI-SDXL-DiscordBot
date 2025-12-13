#!/bin/bash

TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121  # for cuda 12.1
CUDA_VER=12.1

if [ ! -d venv ]; then
    python3.10 -m venv --copies venv
    echo "created new virtualenv"
fi

source venv/bin/activate

PIP_INSTALL_ARGS=()

ROOT_DIR=$(pwd)
EMBEDDED_COMFY_LOCATION="$ROOT_DIR/embedded_comfy"

if [ -f "$ROOT_DIR/requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$ROOT_DIR/requirements.txt")
fi

if [ ! -f config.properties ]; then
    cp config.properties.example config.properties
    echo "copied example config to config.properties"
    echo "add your bot token and comfyui server address to this config"
else
    echo "found existing config.properties; not overwriting"
fi

mkdir -p input out

if [ ! -d "$EMBEDDED_COMFY_LOCATION" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "$EMBEDDED_COMFY_LOCATION"
    echo "created embedded comfy directory"
fi

cd "$EMBEDDED_COMFY_LOCATION"
git pull

if [ -f "$EMBEDDED_COMFY_LOCATION/requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/requirements.txt")
fi

cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyScript ]; then
    git clone https://github.com/Chaoses-Ib/ComfyScript.git
    echo "cloned ComfyScript"
fi
cd ComfyScript
git pull
if [ -f "pyproject.toml" ] || [ -f "setup.py" ] || [ -f "setup.cfg" ]; then
    PIP_INSTALL_ARGS+=("-e" "${EMBEDDED_COMFY_LOCATION}/custom_nodes/ComfyScript[default]")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyUI_Ib_CustomNodes ]; then
    git clone https://github.com/Chaoses-Ib/ComfyUI_Ib_CustomNodes.git
    echo "cloned ComfyUI_Ib_CustomNodes"
fi
cd ComfyUI_Ib_CustomNodes
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/ComfyUI_Ib_CustomNodes/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d was-node-suite-comfyui ]; then
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
    echo "cloned was node suite"
fi
cd was-node-suite-comfyui
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/was-node-suite-comfyui/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyUI_Comfyroll_CustomNodes ]; then
    git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git
    echo "cloned ComfyUI_Comfyroll_CustomNodes"
fi

if [ ! -d ComfyUI-AnimateDiff-Evolved ]; then
  git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
  echo "cloned ComfyUI-AnimateDiff-Evolved"
fi
cd ComfyUI-AnimateDiff-Evolved
git pull


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyUI-VideoHelperSuite ]; then
  git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
  echo "cloned ComfyUI-VideoHelperSuite"
fi
cd ComfyUI-VideoHelperSuite
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyUI-audio ]; then
  git clone https://github.com/handoniumumumum/ComfyUI-audio.git
  echo "cloned ComfyUI-audio"
fi
cd ComfyUI-audio
git pull
# Install ComfyUI-audio requirements separately
pip install -r requirements.txt -U --extra-index-url %TORCH_CUDA_INDEX_URL%


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyUI-Advanced-ControlNet ]; then
  git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
  echo "cloned ComfyUI-Advanced-ControlNet"
fi
cd ComfyUI-Advanced-ControlNet
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/ComfyUI-Advanced-ControlNet/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d comfy_controlnet_preprocessors ]; then
  git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors.git
  echo "cloned comfy_controlnet_preprocessors"
fi
cd comfy_controlnet_preprocessors
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/comfy_controlnet_preprocessors/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyUI_AdvancedRefluxControl ]; then
  git clone https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl.git
  echo "cloned ComfyUI_AdvancedRefluxControl"
fi
cd ComfyUI_AdvancedRefluxControl
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/ComfyUI_AdvancedRefluxControl/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d ComfyUI-GGUF ]; then
  git clone https://github.com/city96/ComfyUI-GGUF.git
  echo "cloned ComfyUI-GGUF"
fi
cd ComfyUI-GGUF
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/ComfyUI-GGUF/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/custom_nodes"
if [ ! -d comfyui_controlnet_aux ]; then
  git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
  echo "cloned comfyui_controlnet_aux"
fi
cd comfyui_controlnet_aux
git pull
if [ -f "requirements.txt" ]; then
    PIP_INSTALL_ARGS+=("-r" "$EMBEDDED_COMFY_LOCATION/custom_nodes/comfyui_controlnet_aux/requirements.txt")
fi


cd "$EMBEDDED_COMFY_LOCATION/models/checkpoints"
mkdir -p xl 15 cascade pony svd

cd "$EMBEDDED_COMFY_LOCATION/models/loras"
mkdir -p xl 15 cascade pony

cd "$EMBEDDED_COMFY_LOCATION/models/controlnet"
mkdir -p xl 15 cascade pony

cd "$EMBEDDED_COMFY_LOCATION"
if [ ${#PIP_INSTALL_ARGS[@]} -gt 0 ]; then
    echo "Installing Python requirements..."
    pip install -U "${PIP_INSTALL_ARGS[@]}" --extra-index-url "$TORCH_CUDA_INDEX_URL"
else
    echo "No Python requirements collected for installation."
fi

python main.py --quick-test-for-ci
