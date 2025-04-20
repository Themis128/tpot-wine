# Summary of 32_CatBoost

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.15
- **depth**: 6
- **rsm**: 0.8
- **loss_function**: MultiClass
- **eval_metric**: Accuracy
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

90.7 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.958333 |  0.821429 |  0.944444 |  0.941176 |  0.84     |  0.884615 |   0.891304 |    0.898333 |       0.898333 |  0.939442 |
| recall    |  1        |  1        |  0.73913  |  0.695652 |  0.913043 |  1        |   0.891304 |    0.891304 |       0.891304 |  0.939442 |
| f1-score  |  0.978723 |  0.901961 |  0.829268 |  0.8      |  0.875    |  0.938776 |   0.891304 |    0.887288 |       0.887288 |  0.939442 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |   0.891304 |  138        |     138        |  0.939442 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                0 |               23 |                0 |                0 |                0 |                0 |
| Labeled as 3 |                1 |                4 |               17 |                1 |                0 |                0 |
| Labeled as 4 |                0 |                1 |                1 |               16 |                4 |                1 |
| Labeled as 5 |                0 |                0 |                0 |                0 |               21 |                2 |
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
