#parallel -u --link 

#mpirun -np 5 python scripts/fit_bird_june.py --n_days 30 \
#    -p household school company \
#    -d cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 \
#    --n_epochs 10000 \
#    --lr 0.001 \
#    --data_calibrate cases_per_timestep\
#    -w 0.001 --data_path ./data/camden_synth.csv \
#    --june_config configs/bird_june.yaml \
#    --results_path /cosma7/data/dp004/dc-quer1/birds_results/reg_tests_forward \
#    --n_samples_per_epoch 5 \
#    --diff_mode forward \
#    --chunk_size 0 \
#        # cases_by_age_18 cases_by_age_25 cases_by_age_65 cases_by_age_100 \
        
mpirun -np 5 python scripts/fit_bird_june.py \
    --n_days 65 \
    -p all_no_seed \
    -d cuda:0 cuda:1 cuda:2 cuda:3 cuda:4  \
    --n_epochs 10000 \
    --loss LogMSELoss\
    --lr 0.001 \
    --data_calibrate deaths_per_timestep\
    -w 0.0001 \
    --data_path ./data/camden_synth.csv \
    --june_config configs/bird_june.yaml \
    --results_path /cosma7/data/dp004/dc-quer1/birds_results/camden_deaths_w_0.0001 \
    --n_samples_per_epoch 5 \
    --diff_mode forward \
    --chunk_size 0 \
        # cases_by_age_18 cases_by_age_25 cases_by_age_65 cases_by_age_100 \
