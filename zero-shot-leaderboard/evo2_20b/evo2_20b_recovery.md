# evo2_20b -- motif_acc

Max over context modes: left, right_reverse_complement. Modes present on disk: left, left_complement, right_reverse, right_reverse_complement.

| Task | Split | Token Acc | Motif Acc | Best strand |
| :--- | :--- | :--- | :--- | :--- |
| **acceptor_recovery** | test_maize | 0.872 | 0.826 | rc |
|  | test_tomato | 0.464 | 0.339 | fwd |
| **donor_recovery** | test_maize | 0.877 | 0.822 | fwd |
| **tis_recovery** | test_maize | 0.770 | 0.602 | rc |
|  | test_tomato | 0.766 | 0.586 | rc |
| **tts_recovery** | test_maize | 0.686 | 0.453 | fwd |
|  | test_tomato | 0.656 | 0.379 | fwd |

## Per context mode

| Task | Split | Mode | Token Acc | Motif Acc |
| :--- | :--- | :--- | :--- | :--- |
| acceptor_recovery | test_maize | left | 0.505 | 0.370 |
| acceptor_recovery | test_maize | right_reverse_complement | 0.872 | 0.826 |
| acceptor_recovery | test_tomato | left | 0.464 | 0.339 |
| donor_recovery | test_maize | left | 0.877 | 0.822 |
| donor_recovery | test_maize | right_reverse_complement | 0.581 | 0.357 |
| tis_recovery | test_maize | left | 0.441 | 0.140 |
| tis_recovery | test_maize | right_reverse_complement | 0.770 | 0.602 |
| tis_recovery | test_tomato | left | 0.470 | 0.151 |
| tis_recovery | test_tomato | right_reverse_complement | 0.766 | 0.586 |
| tts_recovery | test_maize | left | 0.686 | 0.453 |
| tts_recovery | test_maize | right_reverse_complement | 0.386 | 0.094 |
| tts_recovery | test_tomato | left | 0.656 | 0.379 |
| tts_recovery | test_tomato | right_reverse_complement | 0.415 | 0.089 |

## Incomplete

- `acceptor_recovery` / `test_tomato`: missing right_reverse_complement
- `donor_recovery_test_tomato`: no output at all (array element produced nothing)
