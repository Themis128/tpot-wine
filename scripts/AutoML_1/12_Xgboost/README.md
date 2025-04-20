# Summary of 12_Xgboost

[<< Go back](../README.md)


## Extreme Gradient Boosting (Xgboost)
- **n_jobs**: -1
- **objective**: multi:softprob
- **eta**: 0.1
- **max_depth**: 8
- **min_child_weight**: 1
- **subsample**: 1.0
- **colsample_bytree**: 1.0
- **eval_metric**: accuracy
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

12.6 seconds

### Metric details
|           |   1 |         2 |         3 |         4 |         5 |   6 |   accuracy |   macro avg |   weighted avg |   logloss |
|:----------|----:|----------:|----------:|----------:|----------:|----:|-----------:|------------:|---------------:|----------:|
| precision |   1 |  0.916667 |  0.681818 |  0.681818 |  0.833333 |   1 |   0.855072 |    0.852273 |       0.852273 |   1.08943 |
| recall    |   1 |  0.956522 |  0.652174 |  0.652174 |  0.869565 |   1 |   0.855072 |    0.855072 |       0.855072 |   1.08943 |
| f1-score  |   1 |  0.93617  |  0.666667 |  0.666667 |  0.851064 |   1 |   0.855072 |    0.853428 |       0.853428 |   1.08943 |
| support   |  23 | 23        | 23        | 23        | 23        |  23 |   0.855072 |  138        |     138        |   1.08943 |


## Confusion matrix
|              |   Predicted as 1 |   Predicted as 2 |   Predicted as 3 |   Predicted as 4 |   Predicted as 5 |   Predicted as 6 |
|:-------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Labeled as 1 |               23 |                0 |                0 |                0 |                0 |                0 |
| Labeled as 2 |                0 |               22 |                0 |                1 |                0 |                0 |
| Labeled as 3 |                0 |                2 |               15 |                5 |                1 |                0 |
| Labeled as 4 |                0 |                0 |                5 |               15 |                3 |                0 |
| Labeled as 5 |                0 |                0 |                2 |                1 |               20 |                0 |
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
