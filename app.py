# app.py

import streamlit as st
import torch
import numpy as np
from torchvision.utils import make_grid

# Import both generator classes from our updated model.py
from model import Generatorcifar, Generatorceleb

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Generative AI Platform", page_icon="🎨", layout="wide")

# --- MODEL CONFIGURATION ---

MODELS_CONFIG = {
    "Faces (CelebA)": {
        "path": "models/generator_celeba.pth",
        "class": Generatorceleb,
        "params": {"nz": 100, "ngf": 64, "nc": 3},  # Use params from CelebA training
        "title": "🎨 AI Face Generator",
        "description": "Create unique, realistic portraits. This model was trained on the CelebA dataset of celebrity faces.",
    },
    "Objects & Animals (CIFAR-10)": {
        "path": "models/netG_epoch_35.pth",  # <-- CHANGE THIS FILENAME
        "class": Generatorcifar,
        "params": {"nz": 100, "ngf": 64, "nc": 3},  # Use params from CIFAR training
        "title": "🎨 AI Image Creator",
        "description": "Create novel images of objects and animals. This model was trained on the CIFAR-10 dataset.",
    },
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- DYNAMIC MODEL LOADING ---
@st.cache_resource
def load_generator(model_name):
    """
    Loads the selected generator model based on the name.
    The cache will store each model separately.
    """
    config = MODELS_CONFIG[model_name]
    model_path = config["path"]
    model_class = config["class"]
    model_params = config["params"]

    try:
        model = model_class(
            nz=model_params["nz"], ngf=model_params["ngf"], nc=model_params["nc"]
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(
            f"Model file not found at {model_path}. Please update the path in MODELS_CONFIG."
        )
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None


# --- UI & LOGIC ---
st.sidebar.header("Image Creation Controls 🎨")

# 1. MODEL SELECTION
selected_model_name = st.sidebar.selectbox(
    "Choose a Model", options=list(MODELS_CONFIG.keys())
)

# Get the configuration for the selected model
active_config = MODELS_CONFIG[selected_model_name]

# 2. DYNAMIC UI TEXT
st.title(active_config["title"])
st.write(active_config["description"])

# 3. LOAD THE SELECTED MODEL
generator = load_generator(selected_model_name)

if generator:
    # 4. GENERATION CONTROLS
    st.sidebar.markdown("---")
    seed = st.sidebar.slider(
        "Starting Point (Random Seed)",
        min_value=0,
        max_value=9999,
        value=42,
        help="The same number will always generate the same starting image.",
    )
    st.sidebar.markdown("**Explore the AI's Imagination**")
    num_sliders = 8
    latent_sliders = []
    for i in range(num_sliders):
        slider_val = st.sidebar.slider(f"Control Slider #{i+1}", -3.0, 3.0, 0.0, 0.1)
        latent_sliders.append(slider_val)

    # Main content area
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control Panel")
        generate_button = st.button("Create Image with Sliders", type="primary")
        random_button = st.button("Create a Random Image")

    with col2:
        st.subheader("Your AI-Generated Image")
        image_placeholder = st.empty()

    # 5. IMAGE GENERATION LOGIC (no changes needed here)
    def generate_image(is_random=False):
        with st.spinner("The AI artist is painting... ✨"):
            torch.manual_seed(seed)
            # Use latent dimension size from the active model's config
            latent_dim = active_config["params"]["nz"]
            with torch.no_grad():
                z = torch.randn(1, latent_dim, 1, 1, device=DEVICE)
                if not is_random:
                    for i in range(num_sliders):
                        z[0, i, 0, 0] = latent_sliders[i]

                generated_tensor = generator(z).detach().cpu()
                img_grid = make_grid(generated_tensor, normalize=True, padding=0)
                img_np = img_grid.permute(1, 2, 0).numpy()
                image_placeholder.image(
                    img_np, caption="Generated Image", use_column_width=True
                )

    if generate_button:
        generate_image(is_random=False)
    if random_button:
        generate_image(is_random=True)
    if (
        "last_model" not in st.session_state
        or st.session_state.last_model != selected_model_name
    ):
        st.session_state.last_model = selected_model_name
        generate_image(is_random=False)
