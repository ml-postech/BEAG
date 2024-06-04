GPU=$1
SEED=$2


python BEAG/main.py \
--env_name 'AntMazeBottleneck-v0' \
--test_env_name 'AntMazeBottleneck-eval-v0' \
--action_max 30. \
--max_steps 600 \
--start_planning_epoch 0 \
--n_cycles 15 \
--n_test_rollouts 10 \
--high_future_step 1 \
--subgoal_freq 600 \
--subgoal_scale 12. 12. \
--subgoal_offset 8. 8. \
--low_future_step 150 \
--subgoaltest_threshold 1 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--cutoff 30 \
--n_initial_rollouts 200 \
--n_graph_node 300 \
--low_bound_epsilon 10 \
--gradual_pen 5.0 \
--subgoal_noise_eps 2 \
--n_epochs 200 \
--cuda_num ${GPU} \
--seed ${SEED} \
--eval_interval 10 \
--method 'grid' \
--high_future_p 0.9 \
--high_penalty 0.3 \
--uncertainty 'value' \
--alpha 0.0 \
--beta 0.0 \
--nosubgoal \
--setting 'FIRG' \
--densify_freq 1 \
--eval_coverage \
--eval_coverage_freq 5 \
--eval_coverage_num 10 \
--AGS
