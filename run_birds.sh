parallel -u --link python scripts/fit_bird_june.py --n_days 30 -p household company school university pub \
    -d cuda:{1} --n_epochs 10000 \
    --data_calibrate cases_by_age_18 cases_by_age_25 cases_by_age_65 cases_by_age_100 \
    -w {2} --data_path ./data/camden_westminster_brent_synth.csv --june_config configs/bird_june.yaml \
    --results_path ./3d_w_{2} --n_samples_per_epoch 5 --diff_mode reverse \
    --chunk_size 0 \
    ::: 4 5 6 7 8 ::: 100.0 10.0 1.0 0.1 0.01
