# Summary of 22_LightGBM

[<< Go back](../README.md)


## LightGBM
- **n_jobs**: -1
- **objective**: multiclass
- **num_leaves**: 63
- **learning_rate**: 0.05
- **feature_fraction**: 0.9
- **bagging_fraction**: 1.0
- **min_data_in_leaf**: 20
- **metric**: custom
- **custom_eval_metric_name**: accuracy
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

7.1 seconds

### Metric details
|           |         1 |         2 |        3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|---------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.851852 |  0.869565 |  0.62963 |  0.923077 |  0.875    |  0.958333 |    0.84058 |    0.851243 |       0.851243 |   1.06804 |
| recall    |  1        |  0.869565 |  0.73913 |  0.521739 |  0.913043 |  1        |    0.84058 |    0.84058  |       0.84058  |   1.06804 |
| f1-score  |  0.92     |  0.869565 |  0.68    |  0.666667 |  0.893617 |  0.978723 |    0.84058 |    0.834762 |       0.834762 |   1.06804 |
| support   | 23        | 23        | 23       | 23        | 23        | 23        |    0.84058 |  138        |     138        |   1.06804 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                2 |               20 |                1 |                0 |                0 |                0 |
| Labeled as 3 |                2 |                3 |               17 |                1 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                8 |               12 |                3 |                0 |
| Labeled as 5 |                0 |                0 |                1 |                0 |               21 |                1 |
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
