# MJLab-QS Window Audit

- dataset: `scripts/outputs/mjlab_qs/windows/rerun_a25_native_qs_g1stage4_expertboost_20260527/velocity_flat_unitree_g1/d_qs_core_h16.pt`
- split_ids: `[0]`
- git_sha: `79d2082609e5b8916ad67ad329378905e285a56e`

| quality | split | windows | terminal_window_rate | fall_window_rate | truncation_window_rate | window_reward_sum_mean | action_norm_mean | window_action_norm_mean_p90 | action_rate_norm_mean | window_action_rate_norm_mean_p90 | command_2_abs_max_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_smooth | train | 728 | 0.0000 | 0.0000 | 0.0000 | 0.3562 | 0.1917 | 0.2072 | 0.1212 | 0.1252 | 0.3713 |
| medium | train | 31600 | 0.0000 | 0.0000 | 0.0000 | 1.2688 | 0.3950 | 0.4632 | 0.0210 | 0.0677 | 0.3391 |
| expert | train | 200178 | 0.0000 | 0.0000 | 0.0000 | 1.2966 | 0.3313 | 0.3877 | 0.0198 | 0.0721 | 0.3379 |
| expert_noisy | train | 50381 | 0.0000 | 0.0000 | 0.0000 | 1.2587 | 0.3244 | 0.3797 | 0.0779 | 0.1048 | 0.3130 |
