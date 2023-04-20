#python scripts/fit_bird_june.py \
mpirun -np 5 python scripts/fit_bird_june.py \
    --start_date 2020-03-01 \
    --n_days 70 \
    -p all \
    -d cuda:5 cuda:6 cuda:7 cuda:8 cuda:9 \
    --n_epochs 10000 \
    --loss RelativeError \
    --lr 0.001 \
    --data_calibrate deaths_per_timestep \
    -w 0.0 \
    --data_path ./data/london_deaths.csv \
    --june_config ./configs/bird_policies.yaml \
    --results_path /cosma7/data/dp004/dc-quer1/birds_results/tests_sd_w_0.0_relative \
    --n_samples_per_epoch 5 \
    --diff_mode forward \
    --chunk_size 0 \
