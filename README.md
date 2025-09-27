# One-versus-One Deep Neural Networks for Adversarial Robustness

This repository provides the official implementation of the paper:  

**Revisiting One-versus-One Classification for Adversarial Robustness**  
by Dimitrios Papaioannou, Vasileios Mygdalis, Ioannis Pitas  
([Preprint PDF](./Revisiting_One_versus_One_Classification_for_adversarial_robustness.pdf))

---

## 🔍 Motivation

Deep Neural Networks (DNNs) have achieved remarkable success in visual recognition tasks, but they remain **fragile against adversarial attacks**. Tiny, human-imperceptible perturbations can drastically alter predictions, raising serious security concerns in domains such as face recognition, autonomous driving, and biometric authentication.

Most existing defenses rely on **adversarial training** or complex architectural modifications. While effective, they come with two drawbacks:

1. **High computational cost** during training.  
2. **Accuracy degradation** on clean (unperturbed) data.  

Our work proposes a different path: instead of focusing solely on new defenses, we revisit the **classification scheme itself**.

---

## 💡 Our Approach: One-vs-One (OvO)

In conventional classification, DNNs use a **One-vs-All (OvA)** scheme, where each class is separated by a single hyperplane from all others. This decision space is vulnerable—adversaries can often cross a single boundary with minimal perturbations.

We adopt the **One-vs-One (OvO)** strategy:
- Each pair of classes is distinguished by a dedicated binary classifier.  
- The final decision is aggregated from all pairwise votes.  
- Intuition: *It is harder to fool many classifiers simultaneously than a single one.*

To further strengthen robustness, we combine OvO with **Hyperspherical Class Prototypes (HCP)**, a defense method that shapes compact class boundaries in the feature space.  

The result is a decision surface that is **naturally more resilient** to adversarial noise, while maintaining high accuracy on clean data.

---

## ✨ Key Contributions
- **Simple, natural defense strategy** requiring no specialized training tricks.  
- **Robustness against white-box and black-box attacks** (FGSM, BIM, MIM, PGD).  
- **Compatibility with existing defenses** (e.g., HCP, adversarial training).  
- **Improved trade-off**: robustness without sacrificing clean accuracy.  
- Extensive evaluation on **CIFAR-10, STL-10, MNIST, Fashion-MNIST, SVHN, and BLAZE** datasets.  

---

## 📊 Highlights of Results

| Model        | CIFAR-10 | STL-10 | MNIST | F-MNIST | BLAZE | SVHN |
|--------------|----------|--------|-------|---------|-------|------|
| OvA (baseline) | 93.36%   | 79.26% | 99.35% | 94.60%  | 73.25%| 96.23% |
| OvO           | 94.15%   | 82.49% | 99.37% | 94.43%  | 68.92%| 95.81% |
| OvO + HCP     | **93.49%** | **83.10%** | **99.28%** | **93.58%** | **73.58%** | **96.91%** |

- On **CIFAR-10 under FGSM**, the baseline OvA model drops to ~18% accuracy, while **OvO+HCP maintains ~45–48%**.  
- On **BLAZE**, OvO+HCP significantly outperforms OvA in both clean and adversarial settings.  

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/DimPapaion/One-versus-One-DNN-Adversarial-Robustness.git
cd One-versus-One-DNN-Adversarial-Robustness
pip install -r requirements.txt
