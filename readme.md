# Breast Cancer Classification from Mammogram Images Using Deep Learning Features and Equilibrium

## Project Overview

This project implements a **Multiscale Deep Equilibrium (DEQ) Model** for **breast cancer classification from mammogram images**. The model is designed to detect and classify cancerous versus non-cancerous cases using mammography images from the **DDSM and CBIS-DDSM datasets**.

The approach leverages deep learning features, multiscale convolutional layers, and equilibrium modeling techniques to improve classification performance on both benign and malignant cases.

---

## Dataset

### Dataset Used

We used the **DDSM Mammography dataset** available on Kaggle, which combines images from two sources:

1. **DDSM (Digital Database for Screening Mammography)** – Contains negative (non-cancerous) mammogram images.
2. **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** – Contains positive (cancerous) mammogram images.

### Dataset Composition

- **Total training examples:** 55,890
  - 14% Positive (cancerous) examples
  - 86% Negative (non-cancerous) examples

### Labels

- **Binary Classification:** `label_normal`
  - `0` → Non-cancerous
  - `1` → Cancerous

- **Multiclass Classification:** `label`
  - `0`: Negative (Non-cancerous)
  - `1`: Benign Calcification
  - `2`: Benign Mass
  - `3`: Malignant Calcification
  - `4`: Malignant Mass

### Preprocessing

- **Image resizing:** All images resized to **299x299 pixels**.
- **Negative images:** DDSM tiles resized from 598x598 to 299x299.
- **Positive images:** Extracted Regions of Interest (ROIs) with padding, followed by random cropping, flipping, rotation, and resizing.
- **Storage format:** TFRecords (TensorFlow format for efficient loading).

**Note on data splitting:**

- The original dataset test set contained only mass cases, and the validation set contained only calcification cases.
- For balanced evaluation, we **combine test and validation sets** when testing.

---

## Model Architecture

We implemented a **Multiscale Deep Equilibrium Model (DEQ)** for mammogram classification.

### Key Features:

1. **Multiscale Convolutions:**
   - Three convolutional layers with kernel sizes **3x3, 5x5, 7x7** to capture features at multiple scales.
   - Feature maps are concatenated to create a richer representation.

2. **Deep Equilibrium Model (DEQ) Concepts:**
   - **Fixed Point Iteration:** Iteratively updates predictions until they stabilize.
   - **Implicit Differentiation:** Computes gradients efficiently during backpropagation.
   - **Stopping Criterion:** Iteration stops when predictions converge within a tolerance threshold.

3. **Classification:**
   - Concatenated multiscale features are passed through fully connected layers for **binary classification**.

---

## Training Configuration

- **Dataset Size:** 11,177 images (preprocessed to 299x299)
- **Train-Test Split:** 80% training (≈ 8,941 images), 20% testing (≈ 2,236 images)
- **Batch Size:** 32 (≈ 280 batches per epoch)
- **Epochs:** 3
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Binary Cross-Entropy with logits

### Training Results

| Epoch | Avg. Loss | Final Batch Loss | Time    |
|-------|-----------|-----------------|---------|
| 1     | 0.3743    | 0.276           | 3m 16s  |
| 2     | 0.2847    | 0.257           | 3m 12s  |
| 3     | 0.2623    | 0.0636          | 3m 21s  |

### Evaluation

- **Accuracy on Test Set:** 89.49%

---

## References

1. **Deep Equilibrium Models**: Bai, S., Kolter, J. Z., & Koltun, V. (2019). [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377). Advances in Neural Information Processing Systems 32 (NeurIPS 2019).
2. **Multiscale Deep Equilibrium Models**: Bai, S., Koltun, V., & Kolter, J. Z. (2020). [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656). Advances in Neural Information Processing Systems 33 (NeurIPS 2020).
3. **Deep Implicit Layers Tutorial**: Duvenaud, D., Kolter, J. Z., & Johnson, M. (2020). [Deep Implicit Layers Tutorial](http://implicit-layers-tutorial.org/). Neural Information Processing Systems Tutorial.
4. **DDSM Mammography Dataset**: [DDSM Mammography Dataset on Kaggle](https://www.kaggle.com/datasets/skooch/ddsm-mammography).
5. **Final DDSM Experiment**: [Final DDSM Experiment on Kaggle](https://www.kaggle.com/code/baselanaya/final-ddsm-experiment/).

