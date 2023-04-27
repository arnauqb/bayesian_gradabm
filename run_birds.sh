#python scripts/fit_bird_june.py \
#python scripts/fit_bird_june.py \
#mpirun -np 5 python scripts/fit_bird_june.py \
parallel -u --link python scripts/fit_bird_june.py \
    --start_date 2020-03-01 \
    --n_days 30 \
    -p beta_household beta_company beta_school \
    -d cuda:{1} \
    --n_epochs 10000 \
    --loss LogMSELoss \
    --lr 0.001 \
    --data_calibrate cases_per_timestep \
    -w 0.005 \
    --data_path ./data/camden_synth.csv \
    --june_config ./configs/bird_gradients.yaml \
    --results_path /cosma7/data/dp004/dc-quer1/birds_results/gradients_{3}_w_0.005 \
    --n_samples_per_epoch 5 \
    --diff_mode {2} \
    --gradient_mode {3} \
    --chunk_size 0 \
    --clip_val 1.0 \
    ::: 5 6 7 ::: reverse reverse reverse ::: pathwise-parts score pathwise
