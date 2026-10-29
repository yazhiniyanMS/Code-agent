| Model | Dev /44 | Unseen-v1 /53 | Combined /97 | family_new /24 | Neighbours /38 | Held-out families /66 (expressible 7 · other 59) | Reproduction c2 /40 · c3 /40 | Held-out combinations c2 /60 · c3 /60 | Bench official · clean · robust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | 27 | 2 | **29** | 0/24 | 4/38 | 2/66 (0/7 · 2/59) | 0/40 · 0/40 | 0/60 · 0/60 | 22 · 19 · 20 |
| C2 | 28 | 3 | **31** | 1/24 | 2/38 | 1/66 (0/7 · 1/59) | 26/40 · 0/40 | 32/60 · 0/60 | 21 · 18 · 20 |
| R1 1301 | 30 | 3 | **33** | 4/24 | 2/38 | 2/66 (1/7 · 1/59) | 36/40 · - | 54/60 · - | 23 · 20 · 23 |
| V11-A 1302 | 30 | 1 | **31** | 2/24 | 3/38 | 3/66 (1/7 · 2/59) | 38/40 · - | 51/60 · - | 27 · 24 · 24 |
| V12-P 1301 | 26 | 2 | **28** | 0/24 | 2/38 | 3/66 (1/7 · 2/59) | 38/40 · - | 50/60 · - | 29 · 25 · 27 |
| V12-S1 1301 | 30 | 3 | **33** | 4/24 | 3/38 | 4/66 (2/7 · 2/59) | 37/40 · - | 55/60 · - | 22 · 19 · 22 |
| V12-S1 1302 | 30 | 1 | **31** | 3/24 | 3/38 | 2/66 (0/7 · 2/59) | 38/40 · - | 50/60 · - | 23 · 20 · 21 |

| Model | Lost against round 54 (by category) | Gained against round 54 | Lost / gained against C2 | Category gate (>= 3 fails) |
| --- | --- | --- | --- | --- |
| C2 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 2, 'reasoning': 2, 'coding': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_three_chars, unseen:write:count_capitalized, unseen:write:double_plus_one, unseen:write:sum_at_odd_positions | 0 / 0 | passes |
| R1 1301 | 4: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'instruction_following': 3, 'hallucination': 1}) | 8: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:longest_word, unseen:write:count_capitalized, unseen:write:sum_at_odd_positions | 1 / 3 | fails |
| V11-A 1302 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 3, 'reasoning': 1, 'coding': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:longest_word, unseen:write:sum_at_odd_positions | 2 / 2 | fails |
| V12-P 1301 | 4: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'instruction_following': 2, 'formatting': 1, 'reasoning': 1}) | 3: dev:bugfix:first_negative, dev:bugfix:product, unseen:write:sum_at_odd_positions | 4 / 1 | passes |
| V12-S1 1301 | 1: unseen:write:count_words_longer_than ({'hallucination': 1}) | 5: dev:bugfix:product, dev:write:last_word, dev:write:longest_word, unseen:write:count_capitalized, unseen:write:sum_at_odd_positions | 4 / 6 | passes |
| V12-S1 1302 | 3: dev:bugfix:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 1, 'reasoning': 1, 'hallucination': 1}) | 5: dev:bugfix:first_negative, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, unseen:write:sum_at_odd_positions | 3 / 3 | passes |

| Run | Start | Trained tokens | Steps | Train min | Total min | Peak RAM MB | Checkpoint sha256 | Packed content sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | round 54 | 229,376 | 56 | 166 | 268 | 11311 | `9c1384982c9b` | `cc2f2fa07777` |
| R1 1301 | C2 | 147,456 | 36 | 74 | 130 | 10827 | `0758684e76d6` | `9909c511043b` |
| V11-A 1302 | C2 | 147,456 | 36 | 83 | 228 | 10891 | `c369abd0823f` | `944ded915c01` |
| V12-P 1301 | C2 | 147,456 | 36 | 103 | 207 | 11061 | `12b0dcc3a716` | `8c1cb2d6cf22` |
| V12-S1 1301 | R1 | 28,672 | 7 | 17 | 84 | 10907 | `cbc133372f4f` | `5075d097a657` |
| V12-S1 1302 | A | 32,768 | 8 | 14 | 58 | 10768 | `3fdbf898bc3e` | `9a198bc80668` |
