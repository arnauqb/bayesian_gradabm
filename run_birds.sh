#parallel -u --link 

mpirun -np 2 python scripts/fit_bird_june.py --n_days 30 \
    -p school \
    -d cuda:4 cuda:5  --n_epochs 10000 --lr 0.001 \
    --data_calibrate cases_per_timestep \
    -w 0 --data_path ./data/camden_synth.csv --june_config configs/bird_june.yaml \
    --results_path /cosma7/data/dp004/dc-quer1/birds_results/test_multigpu_multi \
    --n_samples_per_epoch 5 --diff_mode forward \
    --chunk_size 0 \
    #::: cuda:4 cuda:5 ::: forward reverse

#--load_model ./5_samples_w_1.0/saved_models/best_model_0829.pth \
    #cases_by_age_18 cases_by_age_25 cases_by_age_65 cases_by_age_100 \
