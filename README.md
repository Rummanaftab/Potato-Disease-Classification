# 🥔 Potato Disease Classification using CNN

## 🌱 Overview

This project aims to **classify potato leaf diseases** using **Convolutional Neural Networks (CNN)**. By leveraging deep learning, the system detects whether a potato plant is **healthy** or affected by a disease such as **Early Blight** or **Late Blight**.
The model has been trained and evaluated using the **PlantVillage dataset** — a benchmark dataset for plant disease classification.

---

## 📊 Project Workflow

```mermaid
graph TD;
    A[Dataset Collection<br>(PlantVillage)] --> B[Data Preprocessing<br>Resize, Normalize, Split];
    B --> C[Model Architecture<br>(CNN Layers)];
    C --> D[Model Training<br>TensorFlow / Keras];
    D --> E[Evaluation<br>Accuracy, Loss, Confusion Matrix];
    E --> F[Prediction<br>New Leaf Image];
```

---

## 🧠 Model Architecture

The CNN model consists of several convolutional, pooling, and dense layers that progressively learn spatial and texture-based features from leaf images.

| Layer Type           | Details                                          |
| -------------------- | ------------------------------------------------ |
| **Input**            | 256x256x3 RGB Image                              |
| **Conv2D + ReLU**    | 32 filters, 3×3 kernel                           |
| **MaxPooling2D**     | 2×2 pool size                                    |
| **Conv2D + ReLU**    | 64 filters, 3×3 kernel                           |
| **MaxPooling2D**     | 2×2 pool size                                    |
| **Flatten**          | Converts 2D features to 1D vector                |
| **Dense (ReLU)**     | 128 units                                        |
| **Dropout**          | 0.5 rate                                         |
| **Output (Softmax)** | 3 classes — *Healthy, Early Blight, Late Blight* |

---

## 🗂️ Dataset

* **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **Classes:**

  * `Potato___Healthy`
  * `Potato___Early_Blight`
  * `Potato___Late_Blight`
* **Split:**

  * **Training:** 80%
  * **Validation:** 10%
  * **Testing:** 10%

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/potato-disease-classification.git
cd potato-disease-classification
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Training Notebook

Open the Jupyter notebook and run all cells:

```bash
jupyter notebook "Potatoes Disease Classification Training.ipynb"
```

---

## 📈 Model Performance

| Metric                  | Value                               |
| ----------------------- | ----------------------------------- |
| **Training Accuracy**   | ~98%                                |
| **Validation Accuracy** | ~96%                                |
| **Test Accuracy**       | ~95%                                |
| **Loss Curve**          | Smooth convergence after ~20 epochs |

Example confusion matrix:

```
Healthy       ✅✅✅
Early Blight  ✅✅❌
Late Blight   ✅✅✅
```

---

## 🔮 Future Enhancements

* Implement **Transfer Learning** with EfficientNet or ResNet.
* Deploy model using **Streamlit** or **Flask** for real-time predictions.
* Add **Grad-CAM visualizations** for model interpretability.

---

## 📁 Repository Structure

```
📦 Potato-Disease-Classification
 ┣ 🔜 README.md
 ┣ 🔜 requirements.txt
 ┣ 🔜 Potatoes Disease Classification Training.ipynb
 ┣ 📂 dataset/
 ┣ 📂 models/
 ┣ 📂 results/
```
---

### 🧩 Example Model Pipeline Diagram

```plaintext
          +-----------------------------+
          |     Input Image (256x256)   |
          +-------------+---------------+
                        |
                        ▼
          [Conv2D + ReLU + Pooling Layers]
                        |
                        ▼
               [Flatten + Dense Layers]
                        |
                        ▼
          [Dropout + Softmax Output Layer]
                        |
                        ▼
            Predicted Class: "Early Blight"
```

---

> 🚀 “Deep learning for agriculture — empowering farmers with intelligent crop protection.”
