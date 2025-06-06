# Summary of 38_RandomForest

[<< Go back](../README.md)


## Random Forest
- **n_jobs**: -1
- **criterion**: gini
- **max_features**: 0.5
- **min_samples_split**: 20
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

11.2 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.958333 |  0.846154 |  0.772727 |  0.75     |  0.818182 |  0.916667 |   0.847826 |    0.843677 |       0.843677 |  0.772047 |
| recall    |  1        |  0.956522 |  0.73913  |  0.652174 |  0.782609 |  0.956522 |   0.847826 |    0.847826 |       0.847826 |  0.772047 |
| f1-score  |  0.978723 |  0.897959 |  0.755556 |  0.697674 |  0.8      |  0.93617  |   0.847826 |    0.844347 |       0.844347 |  0.772047 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |   0.847826 |  138        |     138        |  0.772047 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                0 |               22 |                0 |                1 |                0 |                0 |
| Labeled as 3 |                1 |                4 |               17 |                1 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                5 |               15 |                3 |                0 |
| Labeled as 5 |                0 |                0 |                0 |                3 |               18 |                2 |
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
