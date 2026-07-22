# Spatiotemporal Deep Network for Alzheimer's Stage Prediction

MRI-based deep learning pipeline for classifying Alzheimer's disease into 4 stages, comparing a PyTorch baseline against a custom TensorFlow/Keras transfer-learning pipeline.

## Classes

- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

## What's in the Notebook

The notebook (`alzheimers-detection-project2.ipynb`) contains two separate pipelines trained on the same 4-class MRI staging task:

### 1. PyTorch Baseline (built on an external base repo)

- Clones [Verbosi7y/ai-alzheimer-detection](https://github.com/Verbosi7y/ai-alzheimer-detection) for its dataset/model/metrics modules.
- Loads Kaggle's preprocessed Alzheimer MRI dataset.
- Stratified 60% train / 20% validation / 20% test split.
- Balances classes in two stages: class-specific augmentation rates (Non-Demented ×1, Very Mild ×1, Mild ×2, Moderate ×5), then ADASYN oversampling (k=5) to bring all classes to roughly equal representation.
- Trains a PyTorch CNN — Adam optimizer, cross-entropy loss, 25 epochs with early stopping (patience 5).

### 2. Custom TensorFlow/Keras Comparison Pipeline

Trains and compares three architectures on the same data:

| Model | Approach |
|---|---|
| Custom CNN | 3 conv blocks (96→192→384 filters) with batch norm, dropout, global average pooling |
| MobileNetV2 | ImageNet-pretrained backbone, frozen then fine-tuned (top 25%) |
| EfficientNetB0 | ImageNet-pretrained backbone, frozen then fine-tuned (top 25%) |

Shared training setup:
- Oversampling to a minimum of 1,200 samples per class, plus computed class weights (capped at 10.0) to handle residual imbalance.
- On-the-fly augmentation via `tf.data`: horizontal/vertical flips, brightness/contrast/saturation jitter, random 90° rotations.
- Label smoothing (0.1), early stopping (patience 12), and learning-rate reduction on plateau.
- Two-phase training: frozen backbone for up to 55 epochs, then fine-tuning the last 25% of backbone layers for up to 15 more epochs at a lower learning rate (1e-5).
- The best model across all three is selected by **macro F1 score** on the test set, then saved along with its class names, classification report, and confusion matrix.

## Dataset

The notebook pulls preprocessed MRI images from Kaggle's `alzheimer_mri_preprocessed_dataset`.

**Note:** the repo also contains `ALZmier dataset.zip`, which holds `alzheimers_disease_data.csv` — a *tabular* clinical/demographic dataset (age, BMI, cholesterol levels, MMSE score, family history, etc.), not MRI images. This file doesn't appear to be used anywhere in the notebook's pipelines. Worth confirming whether it belongs in this repo or was added for a different experiment before publishing.

## Results

No trained metrics or result images are currently committed. Once the Keras pipeline finishes training, it saves `models/best_model_metrics.json` containing test accuracy, macro F1, the full classification report, and confusion matrix — that JSON (and a plotted confusion matrix image) would be worth committing so the README can show real numbers instead of a placeholder.

## Tech Stack

- **PyTorch** — baseline CNN pipeline
- **TensorFlow / Keras** — MobileNetV2 / EfficientNetB0 / custom CNN comparison
- **scikit-learn, imbalanced-learn (ADASYN)** — class balancing and evaluation metrics
- **Google Colab** — primary execution environment

## How to Run

1. Open the notebook in Google Colab (recommended — it clones the base repo and installs dependencies automatically) or run locally with a CUDA GPU for reasonable training time.
2. Run cells top to bottom for the PyTorch baseline (Steps 0–10).
3. Run the later cells for the TensorFlow/Keras comparison pipeline — point `DATASET_DIR` at your local MRI image folders if not running on Kaggle.
