# Summary of 10_Default_ExtraTrees

[<< Go back](../README.md)


## Extra Trees Classifier (Extra Trees)
- **n_jobs**: -1
- **criterion**: gini
- **max_features**: 0.9
- **min_samples_split**: 30
- **max_depth**: 4
- **eval_metric_name**: accuracy
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

9.9 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.958333 |  0.884615 |  0.809524 |  0.8125   |  0.769231 |  0.92     |   0.862319 |    0.859034 |       0.859034 |  0.661875 |
| recall    |  1        |  1        |  0.73913  |  0.565217 |  0.869565 |  1        |   0.862319 |    0.862319 |       0.862319 |  0.661875 |
| f1-score  |  0.978723 |  0.938776 |  0.772727 |  0.666667 |  0.816327 |  0.958333 |   0.862319 |    0.855259 |       0.855259 |  0.661875 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |   0.862319 |  138        |     138        |  0.661875 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                0 |               23 |                0 |                0 |                0 |                0 |
| Labeled as 3 |                1 |                3 |               17 |                2 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                4 |               13 |                6 |                0 |
| Labeled as 5 |                0 |                0 |                0 |                1 |               20 |                2 |
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
