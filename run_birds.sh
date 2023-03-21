#parallel --link python scripts/fit_bird_june.py --n_days 30 -p household company \
#    -d cuda:{2} --n_epochs 1000 --data_calibrate cases_per_timestep -w 0.01 \
#    --data_path ./data/camden_synth.csv --june_config configs/bird_june.yaml \
#    --results_path ./test_results_{1} --n_samples_per_epoch 5 --diff_mode {1} \
#    ::: forward reverse ::: 5 6 

python scripts/fit_bird_june.py --n_days 61 -p all \
    -d cuda:5 --n_epochs 1000 --data_calibrate deaths_per_timestep -w 0.01 \
    --data_path ./data/london_deaths.csv --june_config configs/bird_june.yaml \
    --results_path ./london_results --n_samples_per_epoch 5 --diff_mode forward \
    --chunk_size 0
