CUDA_VISIBLE_DEVICES=1 python main2.py --model_name gauss --config configs/lego_gauss.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --cache_ignore data --trial.ON --trial.body_arch resmlp --pretrained_ckpt Experiments/R2L__blender_lego_SERVER-20221002-145721/weights/ckpt_best.tar --render_only --render_test --testskip 1 --screen --project Test__R2L_W256D88__blender_lego