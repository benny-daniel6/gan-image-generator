# 🎨 Generative AI Platform: Multi-Model Image Synthesis

**[🚀 View the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)** _<-- Replace with your link after deploying!_

## 📖 Project Overview

This project is a complete, end-to-end web application that allows users to generate synthetic images using two distinct Generative Adversarial Networks (GANs). It demonstrates the full lifecycle of a machine learning project, from training and implementation to building an interactive frontend and deploying it as a public-facing tool.

The platform provides a user-friendly interface to switch between a model trained from scratch on the **CIFAR-10** dataset (generating objects and animals) and a high-quality, pre-trained model for the **CelebA** dataset (generating realistic human faces).

---

## ✨ Key Features

- **Dual Model Support:** Seamlessly switch between two different GANs in the UI, each with its own unique architecture and dataset.
- **Interactive Latent Space Exploration:** Use a series of sliders to manipulate the input noise vector, giving users intuitive control over the features of the generated images.
- **Full Training Pipeline:** Includes the complete `train.py` script to train the CIFAR-10 model from scratch, demonstrating a foundational understanding of the GAN training process.
- **Pre-Trained Model Integration:** Shows resourcefulness and practical skills by successfully integrating a powerful, pre-trained model downloaded from a public source (Kaggle).
- **End-to-End MLOps Deployment:** The entire application is version-controlled with Git, handles large files with Git LFS, and is deployed on Hugging Face Spaces with an automated CI/CD workflow.

---

## 🏛️ Architectural Overview

1.  **Backend (`model.py`, `train.py`)**: Two distinct PyTorch DCGAN architectures. One is trained locally on CIFAR-10, while the other is a pre-trained model for CelebA.
2.  **Frontend (`app.py`)**: An interactive and user-friendly web interface built with Streamlit. It dynamically loads the selected model and provides controls for generation.
3.  **Deployment (`Hugging Face Spaces`)**: The application is containerized and deployed publicly, managed through a Git-based workflow for continuous updates.

---

## 🛠️ Technology Stack

- **Backend:** Python, PyTorch
- **Frontend:** Streamlit
- **Deployment:** Hugging Face Spaces, Git, Git LFS

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git and Git LFS

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

### 2. Set Up a Virtual Environment

```bash
# Create and activate the environment
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up the Models

You need two model files in the `models/` directory:

- **CIFAR-10 Model (Train it yourself):**
  Run the training script. For a quick test, you can set `num_epochs` to a low number like 35 inside `train.py`.

  ```bash
  python train.py
  ```

  This will create a file like `models/netG_epoch_35.pth`.

- **CelebA Model (Download):**
  Download the pre-trained model file you got from Kaggle.
  - **Download from:** [**INSERT KAGGLE LINK HERE**]
  - Place the downloaded `.pth` file inside the `models/` directory.

### 5. Update and Run the App

1.  Open `app.py` and update the `MODELS_CONFIG` dictionary to point to the exact filenames of your two model files.
2.  Launch the Streamlit app:
    `bash
streamlit run app.py
`
    The application will open in your browser, ready to use.

---

## Acknowledgments

- The pre-trained model for the CelebA dataset was sourced from Kaggle. [**Optional: Link to the specific Kaggle dataset or user here.**]
