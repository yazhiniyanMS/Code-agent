| Model | Dev /44 | Unseen-v1 /53 | Combined /97 | family_new /24 | Neighbours /38 | Held-out families /66 (expressible 7 · other 59) | Reproduction c2 /40 · c3 /40 | Held-out combinations c2 /60 · c3 /60 | Bench official · clean · robust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 54 | 27 | 2 | **29** | 0/24 | 4/38 | 2/66 (0/7 · 2/59) | 0/40 · 0/40 | 0/60 · 0/60 | 22 · 19 · 20 |
| C2 | 28 | 3 | **31** | 1/24 | 2/38 | 1/66 (0/7 · 1/59) | 26/40 · 0/40 | 32/60 · 0/60 | 21 · 18 · 20 |
| R1 1301 | 30 | 3 | **33** | 4/24 | 2/38 | 2/66 (1/7 · 1/59) | 36/40 · - | 54/60 · - | 23 · 20 · 23 |
| R6 1301 | 26 | 5 | **31** | 3/24 | 1/38 | 2/66 (1/7 · 1/59) | 39/40 · 30/40 | 46/60 · 43/60 | 21 · 18 · 20 |
| R7 1301 | 27 | 2 | **29** | 2/24 | 2/38 | 0/66 (0/7 · 0/59) | 40/40 · - | 55/60 · - | 24 · 21 · 21 |
| V11-A 1302 | 30 | 1 | **31** | 2/24 | 3/38 | 3/66 (1/7 · 2/59) | 38/40 · - | 51/60 · - | 27 · 24 · 24 |
| V11-B 1301 | 29 | 2 | **31** | 1/24 | 3/38 | 2/66 (0/7 · 2/59) | 35/40 · - | 50/60 · - | 26 · 23 · 25 |
| V11-C 1301 | 23 | 2 | **25** | 1/24 | 1/38 | 2/66 (0/7 · 2/59) | 38/40 · - | 51/60 · - | 24 · 20 · 24 |
| V11-W 1301 | 26 | 1 | **27** | 0/24 | 4/38 | 2/66 (0/7 · 2/59) | 21/40 · - | 18/60 · - | 25 · 24 · 23 |

| Model | Lost against round 54 (by category) | Gained against round 54 | Lost / gained against C2 | Category gate (>= 3 fails) |
| --- | --- | --- | --- | --- |
| C2 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 2, 'reasoning': 2, 'coding': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_three_chars, unseen:write:count_capitalized, unseen:write:double_plus_one, unseen:write:sum_at_odd_positions | 0 / 0 | passes |
| R1 1301 | 4: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'instruction_following': 3, 'hallucination': 1}) | 8: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:longest_word, unseen:write:count_capitalized, unseen:write:sum_at_odd_positions | 1 / 3 | fails |
| R6 1301 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:count_greater, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'coding': 1, 'instruction_following': 3, 'reasoning': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:product, dev:write:first_three_chars, unseen:write:count_capitalized, unseen:write:double_plus_one, unseen:write:is_vowel, unseen:write:sum_at_odd_positions | 2 / 2 | fails |
| R7 1301 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, dev:write:starts_with_the, unseen:write:count_words_longer_than ({'instruction_following': 3, 'reasoning': 2}) | 5: dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_three_chars, dev:write:longest_word, unseen:write:count_capitalized | 4 / 2 | fails |
| V11-A 1302 | 5: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than, unseen:write:keys_above ({'instruction_following': 3, 'reasoning': 1, 'coding': 1}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:longest_word, unseen:write:sum_at_odd_positions | 2 / 2 | fails |
| V11-B 1301 | 5: dev:bugfix:remove_spaces, dev:bugfix:sum_of_squares, dev:write:remove_spaces, dev:write:scale_each_by_three, unseen:write:count_words_longer_than ({'instruction_following': 2, 'reasoning': 3}) | 7: dev:bugfix:first_negative, dev:bugfix:index_of_min, dev:bugfix:product, dev:write:first_negative, dev:write:first_three_chars, dev:write:index_of_min, unseen:write:sum_at_odd_positions | 4 / 4 | fails |
| V11-C 1301 | 7: dev:bugfix:longest_word, dev:bugfix:remove_spaces, dev:bugfix:sum_of_squares, dev:write:remove_spaces, dev:write:scale_each_by_three, dev:write:subtract_three_each, unseen:write:count_words_longer_than ({'coding': 1, 'instruction_following': 2, 'hallucination': 2, 'reasoning': 2}) | 3: dev:bugfix:first_negative, dev:bugfix:product, unseen:write:count_capitalized | 7 / 1 | passes |
| V11-W 1301 | 3: dev:bugfix:remove_spaces, dev:write:remove_spaces, unseen:write:count_words_longer_than ({'instruction_following': 1, 'reasoning': 2}) | 1: dev:write:longest_word | 7 / 3 | passes |

| Run | Start | Trained tokens | Steps | Train min | Total min | Peak RAM MB | Checkpoint sha256 | Packed content sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | round 54 | 229,376 | 56 | 166 | 268 | 11311 | `9c1384982c9b` | `cc2f2fa07777` |
| R1 1301 | C2 | 147,456 | 36 | 74 | 130 | 10827 | `0758684e76d6` | `9909c511043b` |
| R6 1301 | C2 | 270,336 | 66 | 128 | 212 | 10810 | `bc429d345b36` | `0e8bf2a6a881` |
| R7 1301 | C2 | 303,104 | 74 | 196 | 326 | 10941 | `0001c192288a` | `2d75e24a5474` |
| V11-A 1302 | C2 | 147,456 | 36 | 83 | 228 | 10891 | `c369abd0823f` | `944ded915c01` |
| V11-B 1301 | C2 | 147,456 | 36 | 82 | 177 | 10954 | `462168ece2ac` | `3861bcb030f1` |
| V11-C 1301 | lmwork | 360,448 | 88 | 224 | 366 | 11316 | `f9cbdd6fe2a0` | `14e953032857` |
| V11-W 1301 | R1 | 0 | 0 | 0 | 91 | 16448 | `0b4f6c8f994b` | - |
