python main2.py --model_name gauss --config configs/fern_gauss.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 16 --datadir_kd data/nerf_synthetic/fern_pseudo --n_pose_video 20 --N_iters 200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --cache_ignore data,__pycache__,torchsearchsorted,imgs --screen --project Gauss__ndc_depth16