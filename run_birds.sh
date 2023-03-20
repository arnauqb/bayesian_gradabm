python scripts/fit_bird_june.py --n_days 60 -p all \
    -d cuda:4 --n_epochs 5 --data_calibrate cases_per_timestep -w 0.01 \
    --data_path ./data/june_synth.csv --june_config configs/bird_june.yaml \
    --results_path ./test_results --n_samples_per_epoch 1

