# Instructions to Run Different Baselines


## SD Inpainting training

see the `sd_inpainting` sub-folder under this folder.

### Installation

Use the same environment as BrushNet.

### Training

```shell
accelerate launch baseline/sd_inpainting/train_sdinpainting.py \
    --pretrained_model_name_or_path krnl/realisticVisionV60B1_v51VAE \
    --output_dir runs/logs/realisticv_sdi_20percent_drop_constant_auto_caption_snr_5_random_flip \
    --train_data_dir data/blenderproc \
    --resolution 512 \
    --seed 42 \
    --learning_rate 1e-5 \
    --train_batch_size 2 \
    --max_train_steps 20000 \
    --tracker_project_name sd_inpainting \
    --report_to wandb \
    --resume_from_checkpoint latest \
    --validation_steps 1000 \
    --checkpointing_steps 5000 \
    --custom_checkpoints 12000 17000 18000 \
    --checkpoints_total_limit 15 \
    --proportion_empty_prompts 0.2 \
    --num_validation_images 5 \
    --caption_column auto_caption \
    --snr_gamma 5 \
    --random_flip
```

### Inference

```shell
accelerate launch baseline/sd_inpainting/test_sdinpainting.py \
    --unet_path runs/logs/realisticv_sdi_20percent_drop_constant_auto_caption_snr_5_random_flip/checkpoint-15000 \
    --csv test.csv \
    --output_dir inference
```

### Evaluation

Refer to the `BrushNet/metrics` foler for evaluating metrics on the inference images folder.

