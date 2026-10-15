# Unseen-task failures of the round-54 checkpoint

Clean dev: 27/44 passed; unseen set: 2/53 passed (robust 1/53).

| Class | Clean dev | Unseen | Total | Share |
| --- | --- | --- | --- | --- |
| incorrect_task_interpretation | 12 | 33 | 45 | 66% |
| algorithmic_or_logic_error | 3 | 9 | 12 | 18% |
| incorrect_code_generation | 0 | 5 | 5 | 7% |
| edge_case_failure | 2 | 2 | 4 | 6% |
| formatting_or_parsing | 0 | 2 | 2 | 3% |
| evaluation_failure | 0 | 0 | 0 | 0% |

## Every clean failure

- `dev:write:first_negative`: **incorrect_task_interpretation**: wrote the body of training function 'loop_smallest_negative'
- `dev:write:index_of_min`: **incorrect_task_interpretation**: wrote the body of training function 'index_of_max'
- `dev:write:is_between`: **edge_case_failure**: agrees with the reference on 83% of probes (is_between(2, 1, 3) gave False (expected True))
- `dev:write:last_word`: **incorrect_task_interpretation**: wrote the body of training function 'second_to_last_word'
- `dev:write:longest_word`: **incorrect_task_interpretation**: wrote the body of training function 'shortest_word'
- `dev:write:product`: **algorithmic_or_logic_error**: agreement with the reference 0% (product([2, 3, 4]) gave [2, 3, 4] (expected 24))
- `dev:write:running_max`: **incorrect_task_interpretation**: computes bench:filter_even instead (agreement with the reference 0.09)
- `dev:write:sum_of_squares`: **incorrect_task_interpretation**: wrote the body of training function 'triangular_number'
- `dev:write:fourth_item`: **incorrect_task_interpretation**: wrote the body of training function 'fifth_item'
- `dev:write:second_to_last_item`: **incorrect_task_interpretation**: wrote the body of training function 'last_item'
- `dev:write:first_three_chars`: **incorrect_task_interpretation**: wrote the body of training function 'first_two_chars'
- `dev:bugfix:first_negative`: **incorrect_task_interpretation**: computes smallest_even instead (agreement with the reference 0.38)
- `dev:bugfix:index_of_min`: **incorrect_task_interpretation**: handed the buggy code back unchanged
- `dev:bugfix:is_between`: **edge_case_failure**: agrees with the reference on 82% of probes (is_between(2, 1, 3) gave False (expected True))
- `dev:bugfix:last_word`: **incorrect_task_interpretation**: wrote the body of training function 'second_to_last_word'
- `dev:bugfix:product`: **algorithmic_or_logic_error**: agreement with the reference 0% (product([2, 3, 4]) gave [2, 3, 4] (expected 24))
- `dev:bugfix:running_max`: **algorithmic_or_logic_error**: agreement with the reference 1% (running_max([1, 3, 2, 5]) gave [None, None, None, None] (expected [1, 3, 3, 5]))
- `unseen:write:tens_digit`: **algorithmic_or_logic_error**: agreement with the reference 1% (tens_digit(47) gave 23.5 (expected 4))
- `unseen:write:last_two_digits`: **incorrect_task_interpretation**: wrote the body of training function 'count_digits'
- `unseen:write:round_down_to_ten`: **incorrect_task_interpretation**: wrote the body of training function 'above_ten'
- `unseen:write:double_plus_one`: **incorrect_task_interpretation**: wrote the body of training function 'double'
- `unseen:write:is_teenager`: **incorrect_code_generation**: NameError: name 'is_scripting' is not defined
- `unseen:write:same_sign`: **incorrect_code_generation**: AttributeError: 'int' object has no attribute 'isnan'
- `unseen:write:last_three`: **incorrect_task_interpretation**: computes filter_odd instead (agreement with the reference 0.14)
- `unseen:write:drop_first`: **incorrect_task_interpretation**: wrote the body of training function 'every_other'
- `unseen:write:swap_ends`: **algorithmic_or_logic_error**: agreement with the reference 1% (swap_ends([1, 2, 3, 4]) gave [[1], [2], [3], [4]] (expected [4, 2, 3, 1]))
- `unseen:write:first_plus_last`: **incorrect_task_interpretation**: wrote the body of training function 'all_even'
- `unseen:write:count_above_mean`: **incorrect_task_interpretation**: wrote the body of training function 'count_nonzero'
- `unseen:write:count_changes`: **incorrect_task_interpretation**: wrote the body of training function 'count_items'
- `unseen:write:chunk_pairs`: **incorrect_task_interpretation**: computes dict_keys instead (agreement with the reference 0.01)
- `unseen:write:count_true_values`: **incorrect_task_interpretation**: computes string_length instead (agreement with the reference 0.01)
- `unseen:write:repeat_word`: **incorrect_task_interpretation**: computes bench:gcd instead (agreement with the reference 0.07)
- `unseen:write:count_doubled_letters`: **incorrect_task_interpretation**: wrote the body of training function 'count_punct_chars'
- `unseen:write:dashes_for_spaces`: **incorrect_task_interpretation**: wrote the body of training function 'dots_to_dashes'
- `unseen:write:longest_word_length`: **incorrect_task_interpretation**: computes bench:count_words instead (agreement with the reference 0.13)
- `unseen:write:words_starting_with_vowel`: **formatting_or_parsing**: syntax error: invalid syntax
- `unseen:write:is_all_lowercase_letters`: **incorrect_task_interpretation**: wrote the body of training function 'is_all_lower'
- `unseen:write:count_letter_case_insensitive`: **incorrect_task_interpretation**: computes unseen:count_words_longer_than instead (agreement with the reference 0.69)
- `unseen:write:alternate_case`: **incorrect_task_interpretation**: wrote the body of training function 'make_cap'
- `unseen:write:sum_at_odd_positions`: **incorrect_task_interpretation**: computes sum_odd instead (agreement with the reference 0.19)
- `unseen:write:second_smallest`: **incorrect_task_interpretation**: wrote the body of training function 'smallest_odd'
- `unseen:write:last_index_of`: **algorithmic_or_logic_error**: agreement with the reference 5% (last_index_of([1, 2, 1], 1) gave 0 (expected 2))
- `unseen:write:insert_at_front`: **incorrect_task_interpretation**: computes positions_of instead (agreement with the reference 0.00)
- `unseen:write:count_strictly_between`: **algorithmic_or_logic_error**: agreement with the reference 46% (count_strictly_between([1, 5, 10, 3], 2, 8) gave 1 (expected 2))
- `unseen:write:is_non_increasing`: **incorrect_task_interpretation**: wrote the body of training function 'longer_than_one'
- `unseen:write:without_last`: **incorrect_code_generation**: NameError: name 'last' is not defined
- `unseen:write:hours_to_seconds`: **incorrect_task_interpretation**: wrote the body of training function 'hours_to_minutes'
- `unseen:write:next_even`: **incorrect_task_interpretation**: wrote the body of training function 'above_zero'
- `unseen:write:digits_of`: **algorithmic_or_logic_error**: agreement with the reference 1% (digits_of(123) gave [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56
- `unseen:write:count_capitalized`: **incorrect_task_interpretation**: computes count_islands_1d instead (agreement with the reference 0.61)
- `unseen:write:average_word_length`: **incorrect_code_generation**: TypeError on average_word_length('a bb ccc'): 'key' is an invalid keyword argument for sum()
- `unseen:write:is_vowel`: **formatting_or_parsing**: syntax error: unmatched ']'
- `unseen:write:swap_halves`: **incorrect_task_interpretation**: wrote the body of training function 'swap_case'
- `unseen:write:step_up_to`: **algorithmic_or_logic_error**: agreement with the reference 41% (step_up_to(5, 5) gave 6 (expected 5))
- `unseen:write:repeat_each`: **incorrect_task_interpretation**: wrote the body of training function 'take'
- `unseen:write:ends_with_vowel`: **edge_case_failure**: agrees with the reference on 76% of probes (ends_with_vowel('banana') gave False (expected True))
- `unseen:write:sum_of_last_two`: **incorrect_task_interpretation**: computes bench:sum_of_evens instead (agreement with the reference 0.07)
- `unseen:write:first_index_above`: **algorithmic_or_logic_error**: agreement with the reference 2% (first_index_above([1, 5, 9], 4) gave 9 (expected 1))
- `unseen:write:middle_of_three`: **incorrect_task_interpretation**: wrote the body of training function 'min_of_three'
- `unseen:write:sum_of_last_n`: **algorithmic_or_logic_error**: agreement with the reference 27% (sum_of_last_n([1, 2, 3, 4], 2) gave 3 (expected 7))
- `unseen:write:contains_digit`: **edge_case_failure**: agrees with the reference on 78% of probes (contains_digit('abc1') gave False (expected True))
- `unseen:write:pad_right`: **incorrect_task_interpretation**: wrote the body of training function 'trim'
- `unseen:write:abbreviate`: **algorithmic_or_logic_error**: agreement with the reference 0% (abbreviate('international', 5) gave 5 (expected 'inter.'))
- `unseen:write:count_increases`: **incorrect_task_interpretation**: wrote the body of training function 'count_negative'
- `unseen:write:same_length`: **incorrect_task_interpretation**: wrote the body of training function 'are_equal'
- `unseen:write:longer_of_two`: **incorrect_code_generation**: AttributeError: 'str' object has no attribute 'to_dict'
- `unseen:write:first_repeated`: **incorrect_task_interpretation**: wrote the body of training function 'every_other'
- `unseen:write:count_distinct`: **incorrect_task_interpretation**: wrote the body of training function 'count_items'
