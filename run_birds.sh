#mpirun -np 5 python scripts/fit_bird_june.py \
python scripts/fit_bird_june.py \
    --start_date 2020-03-01 \
    --n_days 30 \
    -p all \
    -d cpu \
    --n_epochs 10000 \
    --loss LogMSELoss \
    --lr 0.001 \
    --data_calibrate cases_per_timestep \
    -w 0.0 \
    --data_path ./data/camden_synth.csv \
    --june_config ./data/synth_config.yaml \
    --results_path ./tests_sd \
    --n_samples_per_epoch 5 \
    --diff_mode reverse \
    --chunk_size 0 \
    #--load_model /cosma7/data/dp004/dc-quer1/birds_results/london_real_data_w_0.001/saved_models/best_model_1673.pth
    #::: 7 ::: 0.0001
        # cases_by_age_18 cases_by_age_25 cases_by_age_65 cases_by_age_100 \
