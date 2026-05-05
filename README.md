# AutoSMC Paper Implementation and Novelty 

## AutoSMC Paper
https://ieeexplore.ieee.org/document/10556687/
 
## Dataset

This project uses the following RadioML datasets:

* RadioML 2016.10A: https://www.kaggle.com/datasets/sanjeevharge/2016-10a
* RadioML 2016.10B: https://www.kaggle.com/datasets/sramjee/rml10b

---

## Directory Structure

* **all_model_graphs/**
  Contains implementations of AutoSMC and AutoSMC* models, along with other models implemented in the paper for benchmarking.

* **run_variants_outputs/**
  Includes outputs/results from each of the novelty variants explored during experimentation.

* **constellation_images/**
  Contains constellation plots generated for each SNR level using the corresponding IQ samples from the dataset.

* **novelty_codes/**
  Includes complete code implementations for all proposed novelty variants.

---

## Novel Contributions

### 1. Input Pipeline

* Raw IQ Signals combined with Constellation Diagrams
* SimCLR Contrastive Pre-training using NT-Xent Loss

---

### 2. Augmentation Pipeline

* Mixup in IQ Space
* CutMix for Signal Segments
* Frequency Domain Augmentation using Parseval's Theorem

---

### 3. CRFF Block Enhancements

* Integration of Self-Attention inside CRFF

  * Pre-LayerNorm Multi-Head Self-Attention with Residual Connections

---

### 4. NAS Pipeline Improvements

* Differentiable Fusion Search (DARTS) replacing Bayesian Search

Fusion strategies explored:

* Weighted Sum
* MLP-based Fusion
* Cross Attention
* Multi-Head Self-Attention (MHSA)

---

## Notes

This repository focuses on improving modulation classification performance through architectural innovations, enhanced data augmentation strategies, and advanced neural architecture search techniques.
