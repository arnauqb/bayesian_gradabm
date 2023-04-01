#parallel -u --link 

python scripts/fit_bird_june.py --n_days 30 \
    -p all \
    -d cuda:0 --n_epochs 10000 --lr 0.001 \
    --data_calibrate cases_per_timestep \
    -w 0.001 --data_path ./data/camden_synth.csv --june_config configs/bird_june.yaml \
    --results_path /cosma7/data/dp004/dc-quer1/birds_results/tests_w_0.001 \
    --n_samples_per_epoch 5 --diff_mode reverse \
    --chunk_size 0 \
    #::: 4 5 6 7 8 ::: 0.0001 0.001 0.01 0.1 1.0

#--load_model ./5_samples_w_1.0/saved_models/best_model_0829.pth \
    #cases_by_age_18 cases_by_age_25 cases_by_age_65 cases_by_age_100 \
