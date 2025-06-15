| Instance | Optimizer | Regularizer          | Epochs | Early Stopping | Layers           | LR     | Accuracy | F1   | Recall | Precision |
|----------|-----------|-----------------------|--------|----------------|------------------|--------|----------|------|--------|-----------|
| 1  |Adam(Default)   | None                  | 7      | No             | 64-32-16         | Default| 0.95     | 0.95 | 0.93   | 0.96      |
| 2        | Adam      | L2 + Dropout          | 20     | Yes            | 128-64-32        | 0.001  | 0.94     | 0.94 | 0.91   | 0.96      |
| 3        | RMSprop   | L1 + Dropout          | 20     | Yes            | 128-64-32        | 0.001  | 0.91     | 0.91 | 0.88   | 0.94      |
| 4        | SGD    |    L1_L2 + Dropout       | 30     | Yes            | 128-64-32        | 0.01   | 0.90     | 0.89 | 0.86   | 0.93      |
| 5|Logistic Regression| N/A                   | N/A     | N/A           | N/A              | N/A    | 0.95     | 0.95 | 0.93   | 0.96      |
