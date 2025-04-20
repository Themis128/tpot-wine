# Summary of 49_ExtraTrees

[<< Go back](../README.md)


## Extra Trees Classifier (Extra Trees)
- **n_jobs**: -1
- **criterion**: gini
- **max_features**: 0.8
- **min_samples_split**: 40
- **max_depth**: 3
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

12.1 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.92     |  0.758621 |  0.65     |  0.8125   |  0.809524 |  0.851852 |   0.804348 |    0.800416 |       0.800416 |  0.788742 |
| recall    |  1        |  0.956522 |  0.565217 |  0.565217 |  0.73913  |  1        |   0.804348 |    0.804348 |       0.804348 |  0.788742 |
| f1-score  |  0.958333 |  0.846154 |  0.604651 |  0.666667 |  0.772727 |  0.92     |   0.804348 |    0.794755 |       0.794755 |  0.788742 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |   0.804348 |  138        |     138        |  0.788742 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                0 |               22 |                1 |                0 |                0 |                0 |
| Labeled as 3 |                2 |                6 |               13 |                2 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                6 |               13 |                4 |                0 |
| Labeled as 5 |                0 |                1 |                0 |                1 |               17 |                4 |
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
