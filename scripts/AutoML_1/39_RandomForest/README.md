# Summary of 39_RandomForest

[<< Go back](../README.md)


## Random Forest
- **n_jobs**: -1
- **criterion**: gini
- **max_features**: 0.7
- **min_samples_split**: 30
- **max_depth**: 7
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

12.4 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.884615 |  0.875    |  0.75     |  0.809524 |  0.85     |  0.851852 |    0.84058 |    0.836832 |       0.836832 |  0.937267 |
| recall    |  1        |  0.913043 |  0.652174 |  0.73913  |  0.73913  |  1        |    0.84058 |    0.84058  |       0.84058  |  0.937267 |
| f1-score  |  0.938776 |  0.893617 |  0.697674 |  0.772727 |  0.790698 |  0.92     |    0.84058 |    0.835582 |       0.835582 |  0.937267 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |    0.84058 |  138        |     138        |  0.937267 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                1 |               21 |                1 |                0 |                0 |                0 |
| Labeled as 3 |                2 |                3 |               15 |                3 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                3 |               17 |                3 |                0 |
| Labeled as 5 |                0 |                0 |                1 |                1 |               17 |                4 |
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
