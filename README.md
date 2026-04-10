# GAN Project 
This project uses GAN 
# 🎨 Progressive GAN (ProGAN) - Image Generator

## 📌 Project Description

This project implements a **Progressive Growing Generative Adversarial Network (ProGAN)** using PyTorch.
The model generates images starting from **low resolution (4×4)** and progressively increases to **high resolution (256×256)**.

---

## 🚀 Key Features

* Progressive image generation (4 → 256 resolution)
* Alpha blending for smooth transitions
* Custom Generator and Discriminator models
* Automatic image saving during training
* Supports CPU and GPU (CUDA)

---

## 📁 Project Structure

```
genai/
│
├── dataset/
│   └── celeb/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── gan/
│   ├── generator.py
│   ├── discriminator.py
│   └── utils.py
│
├── outputs/
│   └── samples/
│
└── train.py
```

---

## ⚙️ Installation

### 1. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```
pip install torch torchvision
```

---

## ▶️ How to Run

```
python train.py
```

---

## 🧠 How It Works

### 🔹 Generator

* Takes random noise (`Z_DIM = 256`)
* Generates images progressively
* Uses:

  * Pixel Normalization
  * Upsampling layers

### 🔹 Discriminator

* Classifies real vs fake images
* Processes images from high → low resolution

### 🔹 Progressive Training

* Starts at **4×4 resolution**
* Gradually increases to **256×256**
* Uses **alpha blending** for smooth transitions

---

## ⚙️ Configuration

```
Z_DIM = 256
IMAGE_SIZES = [4, 8, 16, 32, 64, 128, 256]
EPOCHS = [2, 2, 2, 2, 2, 2, 2]
```

---

## 🖼️ Output

Generated images are saved in:

```
outputs/samples/
```

Example:

```
step6_256x256_alpha1.00.png
```

---

## ⚠️ Notes

* Low epochs = blurry images
* Higher resolution requires more training
* GPU is recommended for faster training

---

## 🚧 Future Improvements

* Add Gradient Penalty (WGAN-GP)
* Improve training stability
* Increase epochs for better quality
* Upgrade to StyleGAN

---

## 👤 Author

S J Nithika
Acshaya
Evangelien Marira Reji
