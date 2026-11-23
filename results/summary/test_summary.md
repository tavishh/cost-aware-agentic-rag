# Test-set summary: FQI (seed mean)

Each cell: reward difference; * = 95% paired-bootstrap CI excludes 0.

## λ sweep (cost ×1): FQI − threshold

| setting | NQ BM25 | NQ e5 | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|---|
| λ=0.25 | +0.023 | -0.019 | +0.051* | +0.012 | -0.017 |
| λ=0.5 | +0.022 | +0.031* | +0.001 | +0.035 | +0.000 |
| λ=1 | +0.023 | +0.028 | +0.002 | +0.070* | +0.026 |
| λ=2 | +0.011 | +0.036 | +0.000 | +0.031 | +0.018 |

## λ sweep (cost ×1): FQI − best fixed

| setting | NQ BM25 | NQ e5 | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|---|
| λ=0.25 | – | – | – | +0.046* | +0.013 |
| λ=0.5 | – | – | – | +0.149* | +0.072* |
| λ=1 | – | – | – | +0.412* | +0.253* |
| λ=2 | – | – | – | +1.027* | +0.808* |

## Cost sweep with abstention (λ = 0.25): FQI − threshold

| setting | NQ BM25 | NQ e5 | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|---|
| ×0 | +0.070* | – | +0.029* | +0.042* | +0.039* |
| ×0.5 | +0.006 | – | -0.006 | +0.018 | +0.006 |
| ×1 | +0.022 | – | +0.051* | +0.012 | -0.024* |
| ×2 | +0.014 | – | -0.001 | -0.005 | +0.026* |
| ×4 | +0.032* | – | -0.001 | +0.015 | +0.054* |

## Cost sweep with abstention (λ = 0.25): FQI − best fixed

| setting | NQ BM25 | NQ e5 | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|---|
| ×0 | – | – | – | +0.059* | -0.011 |
| ×0.5 | – | – | – | +0.056* | +0.032* |
| ×1 | – | – | – | +0.046* | +0.007 |
| ×2 | – | – | – | +0.056* | +0.044* |
| ×4 | – | – | – | +0.076* | +0.081* |

## Cost sweep without abstention (λ = 0): FQI − threshold

| setting | NQ BM25 | NQ e5 | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|---|
| ×0 | +0.032* | – | +0.035* | +0.027 | +0.046* |
| ×0.5 | -0.004 | – | -0.005 | -0.011 | +0.000 |
| ×1 | +0.002 | – | -0.012 | -0.003 | -0.001 |
| ×2 | -0.000 | – | +0.010* | +0.001 | +0.007 |
| ×4 | -0.005 | – | +0.000 | +0.000 | -0.012 |

## Cost sweep without abstention (λ = 0): FQI − best fixed

| setting | NQ BM25 | NQ e5 | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|---|
| ×0 | – | – | – | +0.015 | -0.016 |
| ×0.5 | – | – | – | -0.016 | +0.008 |
| ×1 | – | – | – | -0.004 | +0.011 |
| ×2 | – | – | – | -0.003 | +0.022 |
| ×4 | – | – | – | +0.000 | -0.007 |

FQI vs threshold over all test comparisons: 15 significant wins, 1 significant losses, out of 60.

'–' = not available (older sweep version without that column, or not run).
