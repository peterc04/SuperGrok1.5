# Matmul-precision analysis — what can stand?

Weights/grads/optimizer state/meta/SAM/eval are fp32 in ALL modes;
this axis changes model fwd/bwd GEMM precision only.
3 seeds per cell; distribution-level comparison (grokking is
trajectory-chaotic, single runs are not evidence).

## fp32
| cell | grok rate | median grok step | final test (mean) | collapse | steps/s |
|---|---|---|---|---|---|
| adamw/decoder | 3/3 | 2600 | 1.000 | 3 | 100.3 |
| grokfast/decoder | 3/3 | 2710 | 1.000 | 2 | 71.9 |
| supergrok15/decoder | 0/3 | — | 0.020 | 2 | 56.7 |
| adamw/vit | 3/3 | 3150 | 0.999 | 3 | 54.6 |

## tf32
| cell | grok rate | median grok step | final test (mean) | collapse | steps/s |
|---|---|---|---|---|---|
| adamw/decoder | 3/3 | 2570 | 1.000 | 3 | 74.7 |
| grokfast/decoder | 3/3 | 2840 | 1.000 | 2 | 69.6 |
| supergrok15/decoder | 0/3 | — | 0.205 | 1 | 51.1 |
| adamw/vit | 2/3 | 3465 | 0.933 | 3 | 67.3 |

## bf16
| cell | grok rate | median grok step | final test (mean) | collapse | steps/s |
|---|---|---|---|---|---|
| adamw/decoder | 3/3 | 2610 | 1.000 | 3 | 69.9 |
| grokfast/decoder | 3/3 | 2430 | 0.999 | 1 | 65.0 |
| supergrok15/decoder | 0/3 | — | 0.079 | 3 | 46.6 |
| adamw/vit | 1/3 | 2640 | 0.490 | 3 | 60.9 |

## fp16amp
| cell | grok rate | median grok step | final test (mean) | collapse | steps/s |
|---|---|---|---|---|---|
| adamw/decoder | 3/3 | 2670 | 1.000 | 3 | 66.3 |
| grokfast/decoder | 3/3 | 2360 | 1.000 | 3 | 53.3 |
| supergrok15/decoder | 1/3 | 1230 | 0.352 | 3 | 31.4 |
| adamw/vit | 1/3 | 2780 | 0.641 | 3 | 44.4 |

## Verdicts (vs fp32 reference)
- **fp32**: 3/4 cells at full grok rate
- **tf32**: 2/4 cells at full grok rate
- **bf16**: 2/4 cells at full grok rate
- **fp16amp**: 2/4 cells at full grok rate