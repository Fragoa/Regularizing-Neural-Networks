# Regularizing Neural Networks on MNIST

This project investigates the use of **L1**, **L2** regularization and **Dropout** to reduce overfitting in neural networks trained on the MNIST dataset.

## Overview

The model experiments were designed in several stages using both a **tiny subset** (5%) and the **full MNIST dataset**. The goal was to find configurations that maximize generalization without overfitting.

---

## 🔬 Experiment Stages

### 1. Baseline Model (M1)
- Architecture: Two hidden layers (8 and 16 neurons)
- Dataset: Tiny set & Full set
- Result: Slight overfitting on the tiny set; decent performance on the full set.

---

### 2. L1 Regularization (M1-L1)
- Tested λ values from `1e-6` to `1e-1`.
- Tiny Set Best λ: `1e-3`
- Full Set Best λ: `1e-5`
- Conclusion: L1 helped reduce overfitting; careful tuning of λ is essential.

---

### 3. L2 Regularization (M1-L2)
- Same λ search procedure.
- L2 provided stable performance with slightly better accuracy in some cases.
- Conclusion: Comparable to L1, but slightly more robust.

---

### 4. Adding Dropout (Model M2)
- Based on best L1 model (λ = `1e-5`)
- Dropout added after each hidden layer
- Tested 22 values for dropout rate `p` from `0.01` to `0.99`.

📌 **Best p (Dropout Rate):** `0.01`  
This setup provided the best balance between validation accuracy and loss, without overfitting.

---

## ✅ Final Results

| Model     | Regularization | λ     | Dropout `p` | Overfitting | Accuracy (Test) |
|-----------|----------------|--------|--------------|--------------|------------------|
| M1        | None           | –      | –            | High         | Moderate         |
| M1-L1     | L1             | 1e-5   | –            | Reduced      | High             |
| M1-L2     | L2             | 1e-5   | –            | Reduced      | High             |
| **M2**    | L1 + Dropout   | 1e-5   | **0.01**     | Low          | **Highest**      |

---

## 📌 Conclusion

- **Best Regularization Setup**: L1 with λ = `1e-5`
- **Best Dropout Rate**: `p = 0.01`
- **Best Overall Model**: M2 — L1 + Dropout  
This combination achieved the highest test accuracy with minimal overfitting.

---

## 📁 Files & Structure

- `model_training.py` — Contains training loops and architecture definitions.
- `dropout_experiments.py` — Code for evaluating dropout performance.
- `plots/` — Accuracy and loss graphs across epochs for all models.
- `results.csv` — Collected metrics for all configurations.

---

## 🔧 Dependencies

- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

Install dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn
