# Released-BiDA-style target OA upper bound

The released BiDA `train_pipeline.py` evaluates Houston18 every 10 epochs and saves the checkpoint with the highest target test OA. This is a target-label oracle selection rule, so this investigation is **separate** from V0's fixed epoch-200 primary results. The BiDA paper reports Houston18 OA 81.11%; the local released-code reproduction reached 82.1553% at epoch 60 under this oracle rule.

We independently rerun A0, A1, and A2 with unchanged 200-epoch training settings, save checkpoints at epochs 10, 20, ..., 200, and evaluate the same 52,901 target pixels after training. The maximum is taken over those 20 candidates, with earliest-epoch tie breaking. Target labels do not affect gradient updates, but they **do** select the reported checkpoint here.

The original V0 trajectory could not be reproduced exactly: source loss drifted by epoch 4–5 in a same-seed audit, including a replay on the original GPU. Rerun results are therefore labeled independent runs. `replay_audit.jsonl` records per-epoch differences from the original history; `epoch_metrics.jsonl` records all 20 target evaluations. The nine completed runs and interpretation are in [RESULTS.md](RESULTS.md).
