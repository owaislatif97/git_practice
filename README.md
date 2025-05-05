# 📦 Wind Power Forecasting Package
Team: PUSH & PRAY
repo: "https://github.com/DTUWindEducation/final-project-push-pray-1"

## Overview
This package provides a comprehensive solution for **short-term wind power forecasting**.  
It is designed to help predict the power output from wind energy production facilities one hour ahead of time, improving grid integration and operational planning.  
It includes modules for **data loading**, **feature engineering**, **model training**, **predictions**, and **evaluation**, all organized into a clean and modular structure for maintainability and scalability.  
The project targets **one-hour ahead** forecasting using Random Forest, Neural Networks, and a Persistence baseline model for performance comparison.
---

## Installation Instructions
To install the package locally, clone the repository and run:

```bash
pip install -e .
```

**Requirements:**
- Python 3.8+
- Install dependencies manually:
  ```bash
  pip install numpy pandas matplotlib scikit-learn pytest pytest-cov pylint
  ```

---

## Package Architecture

Here’s the structure of the package (`src/forecast/`):

```
src/
└── forecast/
    ├── A_data_loader.py        # Load and preprocess raw data
    ├── B_preprocessing.py      # Feature engineering and preprocessing
    ├── C_models.py              # Training Random Forest and Neural Network models
    ├── D_predictor.py           # Prediction interface for trained models
    └── E_evaluation.py          # Evaluation and visualization tools
```

**Diagram:**

```
DataLoader ➔ DataPreprocessor ➔ ModelTrainer ➔ Predictor ➔ Evaluator
```

Each module corresponds to a step in the machine learning pipeline.

---

## Classes Description

| Class | File | Purpose |
|------|------|---------|
| `DataLoader` | `A_data_loader.py` | Loads and cleans CSV weather and power production data. |
| `DataPreprocessor` | `B_preprocessing.py` | Performs feature engineering (lag features, moving averages, time features) and scales data. |
| `ModelTrainer` | `C_models.py` | Trains and tunes Random Forest and Neural Network models with hyperparameter search. |
| `Predictor` | `D_predictor.py` | Makes predictions using trained models or user-provided weather parameters. |
| `Evaluator` | `E_evaluation.py` | Evaluates prediction performance and generates comparison plots. |

---

## Peer Review
- Code was peer-reviewed by classmates during the Round Robins.
- External feedback was used to improve code modularity and documentation.
- Test coverage was implemented to ensure reliable results.

---

 If the reviewer follows the installation instructions, they will be able to **install** and **test** the package immediately.

