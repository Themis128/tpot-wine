# Summary of 39_RandomForest

[<< Go back](../README.md)


## Random Forest
- **n_jobs**: -1
- **criterion**: gini
- **max_features**: 0.5
- **min_samples_split**: 20
- **max_depth**: 4
- **eval_metric_name**: logloss
- **num_class**: 3
- **explain_level**: 0

## Validation
 - **validation_type**: kfold
 - **shuffle**: True
 - **stratify**: True
 - **k_folds**: 5

## Optimized metric
logloss

## Training time

3.7 seconds

### Metric details
|           |      0 |         1 |         2 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|-------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.975 |  0.979167 |  0.944444 |   0.967742 |    0.966204 |       0.968302 |  0.135818 |
| recall    |  0.975 |  0.94     |  1        |   0.967742 |    0.971667 |       0.967742 |  0.135818 |
| f1-score  |  0.975 |  0.959184 |  0.971429 |   0.967742 |    0.968537 |       0.967643 |  0.135818 |
| support   | 40     | 50        | 34        |   0.967742 |  124        |     124        |  0.135818 |


## Confusion matrix
|              |   Predicted as 0 |   Predicted as 1 |   Predicted as 2 |
|:-------------|-----------------:|-----------------:|-----------------:|
| Labeled as 0 |               39 |                1 |                0 |
| Labeled as 1 |                1 |               47 |                2 |
| Labeled as 2 |                0 |                0 |               34 |

## Learning curves
![Learning curves](learning_curves.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Precision Recall Curve

![Precision Recall Curve](precision_recall_curve.png)



[<< Go back](../README.md)
