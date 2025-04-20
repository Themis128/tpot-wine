# Summary of 21_LightGBM

[<< Go back](../README.md)


## LightGBM
- **n_jobs**: -1
- **objective**: multiclass
- **num_leaves**: 63
- **learning_rate**: 0.2
- **feature_fraction**: 0.5
- **bagging_fraction**: 1.0
- **min_data_in_leaf**: 10
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

6.7 seconds

### Metric details
|           |   1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |   1 |  0.846154 |  0.772727 |  0.842105 |  0.807692 |  1        |   0.876812 |    0.878113 |       0.878113 |  0.910963 |
| recall    |   1 |  0.956522 |  0.73913  |  0.695652 |  0.913043 |  0.956522 |   0.876812 |    0.876812 |       0.876812 |  0.910963 |
| f1-score  |   1 |  0.897959 |  0.755556 |  0.761905 |  0.857143 |  0.977778 |   0.876812 |    0.875057 |       0.875057 |  0.910963 |
| support   |  23 | 23        | 23        | 23        | 23        | 23        |   0.876812 |  138        |     138        |  0.910963 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                0 |               22 |                0 |                1 |                0 |                0 |
| Labeled as 3 |                0 |                4 |               17 |                2 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                3 |               16 |                4 |                0 |
| Labeled as 5 |                0 |                0 |                2 |                0 |               21 |                0 |
| Labeled as 6 |                0 |                0 |                0 |                0 |                1 |               22 |

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
