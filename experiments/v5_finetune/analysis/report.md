# Where YCode-LM v5 loses points

## Benchmark (65 items; never trained on, never used to choose)

43 evaluated rounds (8-54), 1927 failed answers in all.

| Category | Points lost, all rounds | Share | Per evaluation, rounds 45-54 (9 evaluations) |
| --- | --- | --- | --- |
| instruction_following | 961 | 50% | 20.8 |
| reasoning | 468 | 24% | 11.6 |
| factual | 195 | 10% | 4.1 |
| coding | 132 | 7% | 2.2 |
| formatting | 87 | 5% | 0.3 |
| hallucination | 84 | 4% | 2.2 |

Most-failed items in those 9 evaluations (failures by category; latest failure):

- `maximum` (standard): failed 9/9 rounds (instruction_following 9); latest: instruction_following, wrote the body of training function 'most_common'
- `minimum` (standard): failed 9/9 rounds (reasoning 5, instruction_following 4); latest: instruction_following, wrote the body of training function 'first_item'
- `average` (standard): failed 9/9 rounds (instruction_following 7, hallucination 2); latest: instruction_following, wrote the body of training function 'sum_nonzero'
- `reverse_string` (standard): failed 9/9 rounds (instruction_following 9); latest: instruction_following, wrote the body of training function 'letters_only'
- `is_palindrome` (standard): failed 9/9 rounds (reasoning 9); latest: reasoning, is_palindrome('racecar') gave False (expected True)
- `factorial` (standard): failed 9/9 rounds (factual 9); latest: factual, wrong result; needs what a factorial is (0! = 1)
- `count_vowels` (standard): failed 9/9 rounds (instruction_following 8, hallucination 1); latest: instruction_following, wrote the body of training function 'string_length'
- `is_prime` (standard): failed 9/9 rounds (instruction_following 5, factual 4); latest: instruction_following, wrote the body of training function 'above_zero'
- `gcd` (standard): failed 9/9 rounds (instruction_following 8, hallucination 1); latest: instruction_following, wrote the body of training function 'larger'
- `celsius_to_fahrenheit` (standard): failed 9/9 rounds (instruction_following 9); latest: instruction_following, wrote the body of training function 'celsius_to_kelvin'
- `unique` (standard): failed 9/9 rounds (instruction_following 8, reasoning 1); latest: reasoning, unique([1, 2, 1, 3, 2]) gave 2 (expected [1, 2, 3])
- `flatten` (standard): failed 9/9 rounds (reasoning 9); latest: reasoning, flatten([[1, 2], [3], []]) gave [[[1, 2], [3], []]] (expected [1, 2, 3])
- `clamp` (standard): failed 9/9 rounds (reasoning 4, instruction_following 4, formatting 1); latest: instruction_following, wrote the body of training function 'power_three'
- `merge_dicts` (standard): failed 9/9 rounds (coding 3, hallucination 3, reasoning 2, instruction_following 1); latest: hallucination, AttributeError: 'dict' object has no attribute 'dict'
- `to_upper` (standard): failed 9/9 rounds (instruction_following 9); latest: instruction_following, wrote the body of training function 'to_lower'

## Dev set (62 items; held out of training, used to accept rounds)

49 evaluated rounds (8-56), 1530 failed answers in all.

| Category | Points lost, all rounds | Share | Per evaluation, rounds 47-56 (10 evaluations) |
| --- | --- | --- | --- |
| instruction_following | 895 | 58% | 13.4 |
| reasoning | 511 | 33% | 9.9 |
| formatting | 79 | 5% | 0.5 |
| hallucination | 26 | 2% | 0.6 |
| coding | 19 | 1% | 0.0 |
| factual | 0 | 0% | 0.0 |

Most-failed items in those 10 evaluations (failures by category; latest failure):

- `double_all` (dev_concept): failed 10/10 rounds (formatting 4, hallucination 3, instruction_following 3); latest: hallucination, NameError: name 'double_d' is not defined
- `first_negative` (dev_concept): failed 10/10 rounds (instruction_following 10); latest: instruction_following, wrote the body of training function 'loop_smallest_negative'
- `is_between` (dev_concept): failed 10/10 rounds (reasoning 10); latest: reasoning, is_between(2, 1, 3) gave False (expected True)
- `last_word` (dev_concept): failed 10/10 rounds (instruction_following 10); latest: instruction_following, wrote the body of training function 'second_to_last_word'
- `longest_word` (dev_concept): failed 10/10 rounds (instruction_following 10); latest: instruction_following, wrote the body of training function 'shortest_word'
- `product` (dev_concept): failed 10/10 rounds (reasoning 8, instruction_following 2); latest: reasoning, product([2, 3, 4]) gave [2, 3, 4] (expected 24)
- `running_max` (dev_concept): failed 10/10 rounds (reasoning 8, instruction_following 2); latest: instruction_following, wrote the body of training function 'keep_odd'
- `smaller` (dev_concept): failed 10/10 rounds (reasoning 9, instruction_following 1); latest: reasoning, smaller(3, 7) gave 10 (expected 3)
- `sum_of_squares` (dev_concept): failed 10/10 rounds (instruction_following 10); latest: instruction_following, wrote the body of training function 'triangular_number'
- `multiply_three_numbers` (dev_slot): failed 10/10 rounds (reasoning 9, instruction_following 1); latest: reasoning, multiply_three_numbers(1, 2, 3) gave 5 (expected 6)
- `fourth_item` (dev_slot): failed 10/10 rounds (instruction_following 10); latest: instruction_following, wrote the body of training function 'fifth_item'
- `second_to_last_item` (dev_slot): failed 10/10 rounds (instruction_following 10); latest: instruction_following, wrote the body of training function 'last_item'
- `first_three_chars` (dev_slot): failed 10/10 rounds (instruction_following 10); latest: instruction_following, wrote the body of training function 'first_two_chars'
- `weeks_to_days` (dev_slot): failed 10/10 rounds (reasoning 10); latest: reasoning, weeks_to_days(1) gave 0.01 (expected 7)
- `double_all` (dev_fix): failed 10/10 rounds (instruction_following 8, hallucination 1, formatting 1); latest: instruction_following, wrote the body of training function 'add_two_each'
