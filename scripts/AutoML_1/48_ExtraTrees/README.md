# Summary of 48_ExtraTrees

[<< Go back](../README.md)


## Extra Trees Classifier (Extra Trees)
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

11.4 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.958333 |  0.793103 |  0.809524 |  0.916667 |  0.72     |  0.851852 |   0.833333 |    0.84158  |       0.84158  |  0.605592 |
| recall    |  1        |  1        |  0.73913  |  0.478261 |  0.782609 |  1        |   0.833333 |    0.833333 |       0.833333 |  0.605592 |
| f1-score  |  0.978723 |  0.884615 |  0.772727 |  0.628571 |  0.75     |  0.92     |   0.833333 |    0.82244  |       0.82244  |  0.605592 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |   0.833333 |  138        |     138        |  0.605592 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                0 |               23 |                0 |                0 |                0 |                0 |
| Labeled as 3 |                1 |                4 |               17 |                1 |                0 |                0 |
| Labeled as 4 |                0 |                1 |                4 |               11 |                7 |                0 |
| Labeled as 5 |                0 |                1 |                0 |                0 |               18 |                4 |
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
