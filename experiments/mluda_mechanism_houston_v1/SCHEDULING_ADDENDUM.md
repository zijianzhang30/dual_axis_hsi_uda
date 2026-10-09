# Scheduling recovery

GPU6/7 tasks M2/M3 seed1174 encountered CUDA allocation errors while moving
the model to GPU, before any optimizer update. Preserve original folders/logs
under failed_before_training. GPU2/3 allocation checks passed; manually restart
only these untrained tasks and adopt unchanged M0/M1 processes. Fixed twelve
scientific runs/seeds/budget, loss and model code are unchanged. New scheduling
GPUs0/1/2/3. No accuracy inspection or algorithm-driven retry. See cuda_recovery.json.
