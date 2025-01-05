# SLURM

To launch a job use the `wrapper.sh`, by running it from the root `BrushNet/` directory. Available commands:

```shell
# ./slurm/wrapper.sh <script_name> <log_file_for_slurm (optional)>

# test run to view gpu and environment info
./slurm/wrapper.sh submit_job_test

# training run
./slurm/wrapper.sh train runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet

# inference run
./slurm/wrapper.sh test runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet

# metrics run
./slurm/wrapper.sh metrics runs/logs/sd15_depth_20percent_drop_constant_auto_caption_train_unet
```

To view logs while job is running, use:
```shell
# To show the last 10 lines of <file> and to wait for <file> to grow:
tail -f <log_file>
```

During the run, log files will be at the root `BrushNet` directory. Once the run is over, they are automatically moved to the `LOG_DIR` passed as input. Default `LOG_DIR` is `slurm/logs`.