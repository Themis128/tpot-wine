# Summary of 9_Default_RandomForest

[<< Go back](../README.md)


## Random Forest
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

10.7 seconds

### Metric details
|           |         1 |         2 |         3 |         4 |         5 |         6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|----------:|
| precision |  0.92     |  0.88     |  0.72     |  0.9      |  0.730769 |  0.851852 |   0.826087 |    0.83377  |       0.83377  |  0.663791 |
| recall    |  1        |  0.956522 |  0.782609 |  0.391304 |  0.826087 |  1        |   0.826087 |    0.826087 |       0.826087 |  0.663791 |
| f1-score  |  0.958333 |  0.916667 |  0.75     |  0.545455 |  0.77551  |  0.92     |   0.826087 |    0.810994 |       0.810994 |  0.663791 |
| support   | 23        | 23        | 23        | 23        | 23        | 23        |   0.826087 |  138        |     138        |  0.663791 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                1 |               22 |                0 |                0 |                0 |                0 |
| Labeled as 3 |                1 |                3 |               18 |                1 |                0 |                0 |
| Labeled as 4 |                0 |                0 |                7 |                9 |                7 |                0 |
| Labeled as 5 |                0 |                0 |                0 |                0 |               19 |                4 |
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
