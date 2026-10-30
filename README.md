# BankRisk-ML

## Notebook setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip ipykernel
python -m ipykernel install --user --name bankrisk-ml --display-name "BankRisk-ML"
```

Open `model.ipynb` in VS Code and select the **BankRisk-ML** kernel.

## Project
Two machine learning modules for finance: loan default prediction and bank complaint classification. Each module trains a simple baseline and a stronger model, then compares them on the same metrics. The focus is sound modeling practice and correct evaluation, not high scores.

## Modules

**1. Loan Default Prediction (tabular)**
- Data: Home Credit Default Risk (`application_train.csv`). About 8% of applicants default.
- Models: logistic regression (baseline), XGBoost, PyTorch feed-forward network.
- Evaluation: ROC-AUC, PR-AUC, and a calibration check. The test set stays untouched until the final comparison. Accuracy is not used.
- Explainability: SHAP summary plot and a single-applicant explanation.

**2. Complaint Classification (NLP)**
- Data: a 20k to 30k row sample of the CFPB Consumer Complaint Database.
- Models: TF-IDF with logistic regression (baseline), fine-tuned DistilBERT.
- Task: classify complaints by product category.
- Evaluation: macro F1 and a confusion matrix.

## Architecture

```
Raw data -> Split (stratified) -> Preprocessing -> Baseline + Stronger model -> Evaluation -> Results
```

## Results

| Module | Model | Metric | Score |
|---|---|---|---|
| Loan default | Logistic regression | ROC-AUC | TBD |
| Loan default | XGBoost | ROC-AUC | TBD |
| Loan default | PyTorch | ROC-AUC | TBD |
| Complaints | TF-IDF + LR | Macro F1 | TBD |
| Complaints | DistilBERT | Macro F1 | TBD |

## Key Findings

TBD after experiments.

## Limitations

TBD after experiments.

## Setup

1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Download `application_train.csv` from the [Home Credit Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk/data) and place it in `data/`.
3. Download the CFPB complaints data from consumerfinance.gov and place the sample in `data/`.
4. Run the notebooks in order.

The `data/` folder is not tracked by Git.

## Repository Structure

- `notebooks/`: one notebook per module
- `models/`: saved models
- `tests/`: PyTest tests
- `data/`: local data (ignored)

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch, Hugging Face Transformers, SHAP, PyTest, matplotlib, Jupyter, Google Colab.