# D+G soup (V6 iteration 6)

Equal-weight average of D (curriculum v1, lr 1.5e-5) and G (same data, lr 3e-5), both fine-tuned
from round 54 with the same seed: `python experiments/v6/soup.py OUT runs/D/<seed>/model_bf16
runs/G/<seed>/model_bf16`, evaluated on dev, unseen-v1, unseen-v2 and the benchmark with
`python -m ycode.lm.cli eval --model OUT --problems {dev,unseen,unseen2,both}`. The composition
(D and G, equal weights) was fixed before the soup's results existed. Seed 1301: all_clean 32/168
(round 54: 29). Seed 1302 replicates it with D and G trained at seed 1302.
