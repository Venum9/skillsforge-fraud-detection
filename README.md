# SkillsForge Fraud Detection Assessment

## Project Overview

This project explores machine learning techniques for detecting fraudulent financial transactions. It was developed as part of the SkillsForge Machine Learning Internship assessment.

The goal is to compare fraud detection models, evaluate their ability to identify suspicious transactions, and investigate how limited fraud analyst review capacity can influence model selection.

## Problem Statement

Fraud detection systems must identify suspicious transactions while minimising unnecessary alerts against legitimate customers. Class imbalance, evolving fraud patterns, delayed fraud labels, and limited investigation resources make this a challenging classification problem.

This project evaluates two machine learning approaches using a public credit card fraud dataset.

## Dataset

- **Total transactions:** 284,807
- **Fraudulent transactions:** 492
- **Legitimate transactions:** 284,315
- **Fraud prevalence:** Approximately 0.173%

The dataset contains anonymised transaction features (`V1`–`V28`), transaction amount, elapsed transaction time, and a fraud label (`Class`).

## Methodology

### Data Preparation

- Sorted transactions by the elapsed-time field.
- Used a chronological split to reduce the risk of training on transactions that occur after the test transactions.
- Used the earliest 80% of transactions for training and the latest 20% for testing.
- Applied class weighting to address the severe imbalance between legitimate and fraudulent transactions.

### Models Evaluated

1. **Logistic Regression** — baseline classification model using feature standardisation and class weighting.
2. **HistGradientBoostingClassifier** — tree-based boosting model trained with balanced sample weights.

### Evaluation Metrics

- Average Precision
- Precision and recall for fraudulent transactions
- Confusion matrix
- False positives and false negatives
- Precision and recall when reviewing the 500 highest-scoring test transactions

## Results

| Metric | Logistic Regression | HistGradientBoosting |
|---|---:|---:|
| Average Precision | 0.7622 | 0.7336 |
| Fraud detected at threshold 0.5 | 68/75 | 63/75 |
| Fraud recall at threshold 0.5 | 90.7% | 84.0% |
| False positives at threshold 0.5 | 1,473 | 150 |
| Fraud cases found in top 500 | 64 | 64 |
| Precision at top 500 | 12.8% | 12.8% |
| Recall at top 500 | 85.3% | 85.3% |

Logistic Regression achieved higher Average Precision and recall at the tested threshold. HistGradientBoosting produced substantially fewer false positives at that threshold. Both models identified 64 fraud cases among the 500 highest-scoring transactions in the test set.

## Repository Contents

- `fraud_detection.ipynb` — Jupyter notebook containing the data preparation, model training, evaluation code, and experiment outputs.
- `README.md` — Project overview, methodology, results, and limitations.
- `technical_report.pdf` — Detailed technical report, if included in this repository.

## Limitations

The test set contained only 75 fraudulent transactions, so performance estimates are uncertain. The elapsed-time field covers a short period and does not provide calendar dates for a daily review-capacity evaluation.

The top-500 experiment represents 500 transactions selected across the entire test period, not 500 transactions per day. The experiment does not establish production readiness, real-world inference latency, or performance on the target company's transaction data.

## Future Improvements

- Evaluate on representative, timestamped payment transaction data.
- Account for delayed fraud labels and prevent data leakage.
- Tune decision thresholds using a separate validation period.
- Evaluate expected financial loss and customer impact.
- Test performance across transaction channels and customer segments.
- Validate daily analyst review capacity and the under-100-millisecond latency requirement.
- Add drift monitoring, model explainability, audit logging, and rollback procedures.

## Disclaimer

This is an experimental machine learning assessment using a public dataset. Results should not be interpreted as evidence of production performance or as financial advice.
