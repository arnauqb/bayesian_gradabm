python scripts/fit_bird_june.py --n_days 15 -p household school company \
    -d cpu --n_epochs 5 --data_calibrate cases_per_timestep -w 0.01 \
    --data_path ./data/june_synth.csv --june_config configs/bird_june_simple.yaml \
    --results_path ./test_results --n_samples_per_epoch 5

