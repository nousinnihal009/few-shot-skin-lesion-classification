# Few-Shot Learning for Rare Skin Disease Classification  

This project implements **Few-Shot Learning** using **Prototypical Networks** to classify **rare skin lesion types** with very few training examples.  

It is designed for the **ISIC 2019 dataset**, where some lesion types have **limited labeled images**.  
By using **meta-learning**, the model can learn new classes with minimal data.  

---

## 📌 Why Few-Shot Learning?  

In medical imaging, many diseases are **rare**, making it hard to collect large labeled datasets.  
Traditional deep learning models need thousands of images, but **few-shot learning** can:  

✅ Learn new lesion categories with **1-5 images**  
✅ Generalize better to **unseen diseases**  
✅ Reduce annotation cost for rare medical conditions  

---

## 🩺 Dataset: ISIC 2019  

We use the **ISIC 2019 Skin Lesion Dataset**, which contains **25,000 dermoscopic images** across 8 lesion types:  

- **MEL** (Melanoma)  
- **NV** (Nevus)  
- **BCC** (Basal Cell Carcinoma)  
- **AK** (Actinic Keratosis)  
- **BKL** (Benign Keratosis)  
- **DF** (Dermatofibroma) *(Rare)*  
- **VASC** (Vascular lesions) *(Rare)*  
- **SCC** (Squamous Cell Carcinoma)  

We split into:  
- **Base classes (for meta-training):** MEL, NV, BCC, AK, BKL  
- **Novel classes (for meta-testing):** DF, VASC  

---

## 🧠 Approach  

We implement a **Prototypical Network**, which learns an **embedding space** where each class is represented by a **prototype vector** (mean of support embeddings).  

For each query image:  
1. Extract embeddings using a CNN backbone (ResNet-18)  
2. Compute **Euclidean distance** to each class prototype  
3. Predict the closest class  

Training uses **episodic meta-learning**:  
- **N-way K-shot tasks**  
- For example, 5-way 1-shot means: 5 classes, 1 support image per class  

---

## 🚀 Installation  

### 1️⃣ Clone this repo  
```bash
git clone https://github.com/<your-username>/few-shot-skin-lesion-classification.git
cd few-shot-skin-lesion-classification
