# Summary of 4_Linear

[<< Go back](../README.md)


## Logistic Regression (Linear)
- **n_jobs**: -1
- **num_class**: 6
- **explain_level**: 0

## Validation
 - **validation_type**: kfold
 - **shuffle**: True
 - **stratify**: True
 - **k_folds**: 10

## Optimized metric
accuracy

## Training time

7.8 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.884615 |  0.875    |  0.666667 |  0.705882 |  0.826087 |  0.958333 |   0.826087 |    0.819431 |       0.819431 |  0.506534 |
| recall    |  1        |  0.913043 |  0.695652 |  0.521739 |  0.826087 |  1        |   0.826087 |    0.826087 |       0.826087 |  0.506534 |
| f1-score  |  0.938776 |  0.893617 |  0.680851 |  0.6      |  0.826087 |  0.978723 |   0.826087 |    0.819676 |       0.819676 |  0.506534 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |   0.826087 |  138        |     138        |  0.506534 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                1 |               21 |                0 |                1 |                0 |                0 |
| Labeled as 3 |                2 |                3 |               16 |                2 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                7 |               12 |                4 |                0 |
| Labeled as 5 |                0 |                0 |                1 |                2 |               19 |                1 |
| Labeled as 6 |                0 |                0 |                0 |                0 |                0 |               23 |

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
