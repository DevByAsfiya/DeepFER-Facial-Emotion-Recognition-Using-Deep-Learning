# DeepFER: Facial Emotion Recognition Using Deep Learning

An end-to-end computer vision system built to classify human facial expressions into seven emotion categories (*angry, disgust, fear, happy, neutral, sad, and surprise*) from static images and live video streams. 

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=flat&logo=github)](https://github.com/DevByAsfiya/DeepFER-Facial-Emotion-Recognition-Using-Deep-Learning)

---

## Overview & Problem Statement

Facial Emotion Recognition (FER) is a fine-grained image classification challenge. The seven emotion classes share the same underlying object (a human face) and differ only through subtle, localised deformations such as brow angles, lip curvature, and eye openings. 

* **Dataset:** Built on the widely used FER-2013 benchmark, comprising 48×48 monochrome face crops (~28.7k training images, ~7.2k validation images). 
* **Challenges Addressed:** Severe class imbalance (the dominant class is roughly 19x larger than the rarest class), label noise, low native resolution (48×48 pixels), and wide variations in lighting, pose, and ethnicity.
* **Business Context:** Real-time affect analysis provides continuous, granular feedback for customer-service interactions, user-experience testing, mental health monitoring, and accessible human-computer interfaces without relying on slow post-interaction surveys.

---

## Key Features & Data Pipeline

1. **Robust Data Hygiene:** Programmatically identified and removed unreadable/zero-byte files, within-training exact duplicates using perceptual difference hashing (dHash), and train-validation data leakage.
2. **Structural Outlier Detection:** Applied intensity-plus-structure rules to filter corrupted or blank frames while preserving legitimate low-light images.
3. **Exploratory Data Analysis (UBM Framework):** Built 15 logical visualisations covering univariate class distributions, bivariate quality metrics (brightness, contrast, sharpness, JPEG size), and multivariate inter-class similarity heatmaps derived from class-average prototypes.
4. **Statistical Hypothesis Testing:** Conducted tests including chi-square goodness-of-fit for class distribution, one-way ANOVA with Tukey HSD for brightness differences across emotions, and two-sample Kolmogorov-Smirnov tests to check distribution shift between splits.
5. **Resilient Training Infrastructure:** Implemented mixed-precision training (AMP) on T4 GPUs with automatic epoch-level checkpointing to Google Drive to protect against Colab session disconnects.

---

## Model Architectures

The project evaluates three models in increasing order of sophistication:
* **Model 1 — Custom CNN (Baseline):** Trained from scratch on 48×48 inputs to establish a baseline performance floor.
* **Model 2 — Frozen ResNet-18:** Uses an ImageNet-pretrained ResNet-18 as a frozen feature extractor with a newly trained classification head on 96×96 inputs.
* **Model 3 — Fully Fine-Tuned ResNet-18 (Production Candidate):** Fine-tunes the entire ResNet-18 network using discriminative learning rates, class-weighted cross-entropy loss, label smoothing, and a one-cycle learning rate schedule.

---

## Evaluation & Metrics

* **Headline Metric:** **Macro-F1 score** is prioritized over plain accuracy to ensure rare classes (such as *disgust*) are properly evaluated and not masked by majority classes.
* **Validation Strategy:** Evaluated on a stratified held-out test split carved out before training, accompanied by confusion matrices and Grad-CAM visualisations to inspect network attention.

---

## Real-Time Inference Pipeline

The trained production model is wrapped in an OpenCV Haar-cascade face detection pipeline:
1. Detect face bounding box on live camera frame.
2. Crop, convert to grayscale, resize, and normalise.
3. Pass through the classifier to predict emotion probabilities.
4. Overlay bounding box and predicted label in real-time.
5. Exported to TorchScript for deployment readiness.

---

## Getting Started & Replication

To run the complete experiment:
1. Open the project notebook in Google Colab or your local Jupyter environment.
2. Ensure GPU acceleration (e.g., T4) is enabled.
3. Upload the dataset zip file into the workspace environment.
4. Configure toggle flags at the top of the notebook (`QUICK_RUN`, `FRESH_START`, `RUN_WEBCAM_DEMO`) depending on whether you want a rapid smoke-test or a full training run.
5. Execute cells sequentially.

---

## Author

* **Asfiya Mukthasar**
