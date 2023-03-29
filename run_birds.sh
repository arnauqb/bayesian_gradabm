python scripts/fit_bird_june.py --n_days 30 \
    -p household \
    -d cpu --n_epochs 1000 --lr 0.001 \
    --data_calibrate cases_per_timestep \
    -w 1.0 --data_path ./data/camden_synth.csv --june_config configs/bird_june.yaml \
    --results_path ./maf_w_1.0 --n_samples_per_epoch 5 --diff_mode reverse \
    --chunk_size 0 \
    #::: 8 ::: 1.0

#--load_model ./5_samples_w_1.0/saved_models/best_model_0829.pth \
    #cases_by_age_18 cases_by_age_25 cases_by_age_65 cases_by_age_100 \
