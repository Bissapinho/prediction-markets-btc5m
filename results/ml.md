| Model | Setting | Log-loss (model) | Log-loss (market) | Gain (nats) | 95 % CI | p adj. (BH) |
|---|---|---|---|---|---|---|
| Logistic regression | standalone | 0.3864 | 0.3883 | +0.0019 | [+0.0012, +0.0027] | < 0.001 |
| Gradient boosting (HistGB) | standalone | 0.3997 | 0.3883 | -0.0114 | [-0.0146, -0.0084] | 1.000 |
| MLP (64, 32) | standalone | 0.3909 | 0.3883 | -0.0026 | [-0.0042, -0.0012] | 1.000 |
| Market price, recalibrated (logistic on logit p) | standalone | 0.3878 | 0.3883 | +0.0005 | [+0.0002, +0.0008] | 0.009 |
| GRU (30 s sequences + static features) | standalone | 0.3883 | 0.3885 | +0.0002 | [-0.0010, +0.0013] | 0.630 |
| Logistic regression | stacked on market price | 0.3874 | 0.3883 | +0.0008 | [+0.0001, +0.0016] | 0.047 |
| Gradient boosting (HistGB) | stacked on market price | 0.3898 | 0.3883 | -0.0016 | [-0.0029, -0.0004] | 1.000 |
| MLP (64, 32) | stacked on market price | 0.3881 | 0.3883 | +0.0002 | [-0.0006, +0.0010] | 0.630 |
