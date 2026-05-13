# Model Interpretability: Calculating Feature Importance
[![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-blue)](https://scikit-learn.org/)
[![Interpretability](https://img.shields.io/badge/Focus-Model%20Explainability-orange)](https://en.wikipedia.org/wiki/Feature_importance)
[![Library](https://img.shields.io/badge/Library-Scikit--Learn-green)](https://scikit-learn.org/stable/modules/tree.html)

## 🏗️ Project Overview
This repository explores the "Why" behind machine learning predictions. By utilizing **Decision Tree Classifiers** and **Random Forest Ensembles**, this project demonstrates the mathematical calculation of **Feature Importance**. 

Understanding which variables drive the model's decisions is vital for domain expertise, feature selection, and building trust in automated systems.

---

## 🛠️ Technical Implementations

### 1. Decision Tree Splitting Logic
The project starts with a single Decision Tree to visualize how a model partitions data.
*   **Gini Impurity / Entropy:** Calculating the "purity" of nodes to determine the most informative features.
*   **Plot Tree Visualization:** Using `sklearn.tree.plot_tree` to create a visual map of the root nodes and leaf nodes, showing exactly where the first (most important) split occurs.

### 2. Feature Importance Calculation
*   **Impurity Reduction:** Quantifying how much each feature decreases the overall weighted impurity of the model.
*   **Ranking:** Normalizing these values to provide a 0-1 scale of feature influence.

### 3. Random Forest Validation
To ensure the importance scores are robust and not biased by a single tree's architecture, we implement a **Random Forest**:
*   **Ensemble Averaging:** Calculating the mean decrease in impurity across 100+ different trees.
*   **Stability:** Proving that certain features consistently remain high-impact even when the data is bootstrapped and features are randomly sub-sampled.

---

## 📊 Visualizations Included
*   **Tree Architecture:** A full hierarchical map of the Decision Tree logic.
*   **Importance Bar Charts:** Comparative rankings showing which predictors (e.g., age, income, blood pressure) the model relies on most.

---

## 💻 Tech Stack
*   **Language:** Python 3.9+
*   **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib
*   **Environment:** Jupyter Notebook

---

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/feature-importance-interpretability-ml.git](https://github.com/your-username/feature-importance-interpretability-ml.git)

2. **Install Dependencies:**
   ```bash
   pip install scikit-learn pandas matplotlib

3. **Run the Analysis:**
   
   Open `how-feature-importance-is-calculated.ipynb` to walk through the visual and mathematical breakdown of feature selection.   
