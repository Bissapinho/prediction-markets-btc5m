| Model | Setting | Log-loss (model) | Log-loss (market) | Gain (nats) | 95 % CI (day) | p adj. (day) | 95 % CI (market) | p adj. (market) | p adj. (hour) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | standalone | 0.3864 | 0.3883 | +0.0019 | [+0.0011, +0.0027] | < 0.001 | [+0.0012, +0.0027] | < 0.001 | < 0.001 |
| Gradient boosting (HistGB) | standalone | 0.3997 | 0.3883 | -0.0114 | [-0.0144, -0.0085] | 1.000 | [-0.0146, -0.0084] | 1.000 | 1.000 |
| MLP (64, 32) | standalone | 0.3909 | 0.3883 | -0.0026 | [-0.0045, -0.0008] | 1.000 | [-0.0042, -0.0012] | 1.000 | 1.000 |
| Market price, recalibrated (logistic on logit p) | standalone | 0.3878 | 0.3883 | +0.0005 | [+0.0002, +0.0008] | 0.001 | [+0.0002, +0.0008] | 0.009 | 0.008 |
| GRU (30 s sequences + static features) | standalone | 0.3883 | 0.3885 | +0.0002 | [-0.0014, +0.0017] | 0.670 | [-0.0010, +0.0013] | 0.630 | 0.639 |
| Logistic regression | stacked on market price | 0.3874 | 0.3883 | +0.0008 | [+0.0001, +0.0016] | 0.039 | [+0.0001, +0.0016] | 0.047 | 0.075 |
| Gradient boosting (HistGB) | stacked on market price | 0.3898 | 0.3883 | -0.0016 | [-0.0028, -0.0003] | 1.000 | [-0.0029, -0.0004] | 1.000 | 1.000 |
| MLP (64, 32) | stacked on market price | 0.3881 | 0.3883 | +0.0002 | [-0.0006, +0.0009] | 0.616 | [-0.0006, +0.0010] | 0.630 | 0.639 |
