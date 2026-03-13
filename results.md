# Model Results – Credit Card Fraud Detection

## Problem Context

Credit card fraud detection is a highly imbalanced classification problem.  
Fraudulent transactions represent less than 1% of all transactions, making traditional accuracy metrics misleading.

The goal of this project was to build a model that maximizes **fraud recall** while maintaining reasonable precision.

---

## Models Evaluated

### Logistic Regression

- Precision (Fraud): ~0.70  
- Recall (Fraud): ~0.83  
- F1-score (Fraud): ~0.76  

### Random Forest (Threshold Tuned)

- Precision (Fraud): ~0.70  
- Recall (Fraud): ~0.89  
- F1-score (Fraud): ~0.78  

Random Forest performed better due to its ability to capture nonlinear patterns and interactions in transaction features.

---

## Confusion Matrix (Random Forest – Threshold = 0.3)

- True Negatives (TN): 56,827  
- False Positives (FP): 37  
- False Negatives (FN): 11  
- True Positives (TP): 87  

---

## Interpretation of Results

- The model successfully detects the majority of fraudulent transactions.
- Very low **false negatives (FN)** are critical in fraud detection, since missed fraud leads to direct financial loss.
- Some **false positives (FP)** are acceptable because flagged transactions can be manually reviewed.
- Precision decreases as recall increases — this reflects the real-world tradeoff in fraud detection systems.
- Threshold tuning allowed the model to prioritize fraud detection performance over raw accuracy.

---

## Key Takeaways

- Handling class imbalance is essential in real-world ML problems.
- Precision–Recall evaluation is more informative than accuracy for rare-event detection.
- Random Forest provides strong baseline performance for tabular fraud detection tasks.
- Model threshold tuning significantly improves business-relevant outcomes.

---

## Future Improvements

- Try advanced imbalance techniques (SMOTE, Balanced Random Forest)
- Experiment with Gradient Boosting / XGBoost
- Build real-time fraud scoring API
- Perform hyperparameter tuning