# Benchmark: Generative Models for Image Reconstruction (PyTorch vs MONAI)

This repository contains a comprehensive comparative study of generative neural networks. The goal is to evaluate reconstruction fidelity and latent space smoothness using four different architectures, progressing from standard AutoEncoders to state-of-the-art Vector Quantized models.

## Technical Context: Why MONAI for MNIST?
Although MNIST consists of handwritten digits, the **MONAI framework** (Medical Open Network for AI) is utilized here to demonstrate high-performance architectural blocks usually reserved for 3D medical imaging (MRI, CT scans). By benchmarking these models on MNIST, we validate the architecture's ability to compress and reconstruct structural information before scaling to complex volumetric data.

## The 4-Stage Comparison

### 1. 01_PyTorch_Vanilla_AE.ipynb
The baseline model. It focuses on purely deterministic dimensionality reduction.
* **Architecture**: Simple fully connected layers.
* **Key Finding**: Fast convergence but lacks generative capabilities (latent space is not continuous).

### 2. 02_PyTorch_Variational_VAE.ipynb
Introduction of probabilistic modeling.
* **Mechanism**: Reparameterization trick (z = mu + sigma * epsilon).
* **Contribution**: Enforces a Gaussian distribution in the latent space, allowing for smooth digit interpolation (e.g., seeing a 0 gradually turn into a 9).

### 3. 03_MONAI_Medical_VAE.ipynb
Leveraging MONAI's specialized layers.
* **Logic**: Uses medical-optimized convolutional blocks and residual connections.
* **Performance**: Shows higher structural integrity in the reconstruction compared to standard PyTorch implementations.

### 4. 04_MONAI_Advanced_VQVAE.ipynb
The most advanced model in this study.
* **Technology**: Replaces continuous latent variables with a discrete codebook.
* **Advantage**: Solves the "blurry image" problem inherent to traditional VAEs. It produces the sharpest reconstructions and is the backbone of modern generative AI like DALL-E.

## Libraries and Frameworks
* **Deep Learning**: PyTorch, MONAI, TensorFlow.
* **Model Analysis**: Torchinfo (architectural visualization).
* **Data Processing**: NumPy, Pandas.
* **Computer Vision**: OpenCV, Pillow, Matplotlib.
* **Evaluation**: Scikit-Learn.

## Technical Skills
* **Architecture Design**: Implementing AE, VAE, and VQ-VAE from scratch.
* **Framework Mastery**: Bridging standard PyTorch with domain-specific MONAI layers.
* **Advanced Mathematics**: Implementation of Kullback-Leibler Divergence (KLD) and Vector Quantization.
* **Generative AI**: Latent space manipulation and smooth manifold interpolation.
* **Scientific Benchmarking**: Systematic comparison of model performance and reconstruction fidelity.

## Metrics and Visualization
For each model, the training was monitored through:
1. **Reconstruction Loss**: Evaluating pixel-wise accuracy (MSE Loss).
2. **KLD Divergence**: Measuring the regularization of the latent space.
3. **Latent Interpolation**: Visualized through mnist_interpolation.gif to test the generative continuity.

## Conclusion
The benchmark reveals that while standard VAEs provide good distribution, the MONAI VQ-VAE offers superior reconstruction clarity, making it the most suitable architecture for high-precision tasks such as feature extraction in medical or security-sensitive image analysis.
