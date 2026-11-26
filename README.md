# Model Interpretability: How Feature Importance Is Calculated

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-blue)
![Scikit Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Focus](https://img.shields.io/badge/Focus-Model%20Interpretability-green)
![Notebook](https://img.shields.io/badge/Format-Jupyter%20Notebook-orange)

## Overview

Machine learning models can make accurate predictions, but understanding **why a model makes a particular prediction** is equally important.

This project explores how feature importance is calculated using **Decision Tree** and **Random Forest** classifiers. It focuses on understanding how individual features contribute to model decisions, how impurity reduction is used to calculate importance scores, and how feature importance can be examined through visualizations and data perturbation.

The project is designed as a practical learning resource for understanding model interpretability using Python and Scikit-Learn.

## Objectives

- Understand why feature importance is useful in machine learning.
- Explore how Decision Trees select features during splitting.
- Understand Gini impurity and Entropy as measures of node impurity.
- Learn how impurity reduction contributes to feature importance.
- Examine feature importance in Random Forest models.
- Visualize feature rankings using bar charts.
- Explore how changing input data can help investigate model behavior.

## Concepts Covered

### 1. Decision Tree and Feature Splitting

A Decision Tree makes predictions by repeatedly splitting the dataset based on feature values.

At every decision node, the algorithm evaluates possible splits and selects one that improves the purity of the resulting child nodes.

Two commonly used impurity measures are:

**Gini Impurity**

\[
Gini = 1-\sum_{i=1}^{C}p_i^2
\]

**Entropy**

\[
Entropy = -\sum_{i=1}^{C}p_i\log_2(p_i)
\]

Here:
- \(C\) represents the number of classes.
- \(p_i\) represents the proportion of samples belonging to class \(i\).

A lower impurity indicates that the node contains a more homogeneous collection of samples.

### 2. Mean Decrease in Impurity (MDI)

Tree-based models calculate feature importance by measuring how much a feature contributes to reducing impurity across the decision tree.

For a split node, the weighted impurity decrease can be represented as:

\[
\Delta I =
\frac{N_t}{N}
\left[
I_t-
\frac{N_{left}}{N_t}I_{left}-
\frac{N_{right}}{N_t}I_{right}
\right]
\]

Where:
- \(N\) is the total number of training samples.
- \(N_t\) is the number of samples reaching the current node.
- \(N_{left}\) and \(N_{right}\) are the samples reaching the child nodes.
- \(I_t\), \(I_{left}\), and \(I_{right}\) represent their impurity values.

The contributions from splits using a particular feature are accumulated and normalized to obtain its feature importance score.

In Scikit-Learn, tree-based estimators expose these scores through:

```python
model.feature_importances_
```

**Important:** These scores represent relative impurity-based importance within the fitted model. They should not be interpreted as probabilities, causal effects, or proof that a feature independently causes a prediction.

### 3. Random Forest Feature Importance

A Random Forest combines multiple Decision Trees to produce an ensemble prediction.

Each tree is trained using randomized samples and feature selection. Feature importance can then be aggregated across the trees.

This provides an ensemble-level view of which features contribute to impurity reduction.

The project explores how feature importance can be examined using Random Forest models rather than relying only on a single Decision Tree.

### 4. Feature Importance Visualization

Visualizations make feature importance easier to interpret.

The project explores:
- Decision Tree structure using tree visualization.
- Feature importance rankings.
- Bar charts for comparing the relative importance of different features.

These visualizations help identify which variables the fitted model relies on most when making predictions.

### 5. Data Perturbation and Model Interpretation

Changing or perturbing input data can provide another perspective on model behavior.

By examining how a model responds when feature values are modified, we can investigate the relationship between input variables and predictions.

This is useful for developing intuition about model behavior, although perturbation results depend on the chosen method and data distribution.

## Project Workflow

```text
Dataset
   |
   v
Data Preparation
   |
   v
Decision Tree Classifier
   |
   v
Understand Impurity and Splitting
   |
   v
Calculate Feature Importance
   |
   v
Visualize Tree and Feature Rankings
   |
   v
Random Forest Classifier
   |
   v
Compare and Investigate Feature Contributions
   |
   v
Data Perturbation Analysis
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data handling and analysis |
| NumPy | Numerical operations |
| Scikit-Learn | Decision Trees, Random Forests, and feature importance |
| Matplotlib | Data visualization |
| Jupyter Notebook | Interactive experimentation |

## Repository Structure

```text
how-feature-importance-is-calculated/
│
├── how-feature-importance-is-calculated.ipynb
│
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/how-feature-importance-is-calculated.git
```

### 2. Navigate to the Project

```bash
cd how-feature-importance-is-calculated
```

### 3. Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

### 4. Run the Notebook

```bash
jupyter notebook
```

Open:

`how-feature-importance-is-calculated.ipynb`

Run the notebook cells sequentially to explore the implementation and visualizations.

## Key Learning Outcomes

After working through this project, you should be able to:

- Explain how Decision Trees choose splitting features.
- Understand Gini impurity and Entropy.
- Describe how impurity reduction contributes to feature importance.
- Retrieve feature importance scores from Scikit-Learn tree-based models.
- Understand how Random Forest aggregates information across multiple trees.
- Interpret feature importance visualizations.
- Recognize the limitations of impurity-based feature importance.

## Important Considerations

Feature importance is useful, but it has limitations:

- **Correlation:** Highly correlated features can distribute or mask each other's importance.
- **Feature bias:** Impurity-based importance can favor features with many possible split points or categories.
- **Model dependence:** Importance scores depend on the model, training data, and selected parameters.
- **Not causation:** A high importance score does not establish a causal relationship.
- **Validation:** Permutation importance on unseen validation data can provide a complementary perspective.

For more reliable interpretation, feature importance should be considered alongside domain knowledge, validation performance, and other explainability techniques.

## Future Improvements

Potential extensions include:

- Comparing impurity-based importance with permutation importance.
- Exploring SHAP values for local and global explanations.
- Studying the impact of correlated features.
- Comparing feature rankings across multiple models.
- Evaluating feature importance on unseen test data.

## Author

**Mani Ratan**

- GitHub: [@mani787060](https://github.com/mani787060)
- LinkedIn: [Mani Ratan](https://linkedin.com/in/mani-ratan-159963289)
- Kaggle: [mani787060](https://www.kaggle.com/mani787060)

---

**The goal of this project is to understand not only how a machine learning model predicts, but also which features influence its decisions and how those contributions are measured.**
