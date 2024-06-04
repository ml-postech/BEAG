GPU=$1
SEED=$2



python BEAG/main.py \
--env_name 'AntMazeComplex-v0' \
--test_env_name 'AntMazeComplex-v0' \
--action_max 30. \
--max_steps 2000 \
--high_future_step 1 \
--subgoal_freq 2000 \
--subgoal_scale 28. 28. \
--subgoal_offset 24. 24. \
--low_future_step 150 \
--subgoaltest_threshold 1 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--cutoff 30 \
--n_initial_rollouts 700 \
--n_graph_node 500 \
--low_bound_epsilon 10 \
--gradual_pen 5.0 \
--subgoal_noise_eps 2 \
--n_epochs 100 \
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
--eval_visualization \
--AGS
