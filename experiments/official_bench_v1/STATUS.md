# Official benchmark validation status

## HyRANK

Started 2026-10-02 on GPUs 6 and 7: A/B/C × seeds 2100/2101/2102.
Each task trains 200 epochs, saves checkpoints every 10 epochs, and freezes
epoch-200 predictions. Target scoring is deferred until all nine predictions
are frozen. Target accuracy never selects checkpoints or changes training.

- A: complete released BiDA optimization loop, native losses, branches and EMA.
- B: BiDA-self + Full Joint BN + original tokenizer, source CE only.
- C: BiDA-self + Full Joint BN + center-relative relation tokenizer, source CE only.

The implementation checks passed for all nine model initializations, including
finite logits/gradients and nonzero target gradients through joint statistics
in B/C. The relation tokenizer is adapted only from 48 to 176 input bands.

Data: `/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK/`.
Author screened `Dioni_gt_out68.mat` and `Loukia_gt_out68.mat` are used.
Target nonzero GT masks supply candidate centers; class values do not enter
training. Native boundary filtering and source split are recorded in manifests.

Results: `results/official_bench_v1/` links to the NAS directory of the same name.
Follow `launcher_status.json`, `launcher.log`, and `hyrank_<arm>_<seed>.log`.
The launcher automatically writes `summary.json` and `RESULTS.md` after all jobs
finish successfully and data/batch/RNG consistency checks pass. No follow-up
experiments or automatic model selection are performed.

## Other benchmarks

MFF is deferred at the user's request. No verified author SD/TD1/TD2 data or
split was found; no alternate Mengjiagang dataset has been substituted.
Author public data repository: https://github.com/YuxiangZhang-BIT/Data-CSHSI

Houston historical B/C results exist, but historical complete-BiDA training
does not strictly match the released loss arithmetic order and the current
source-validation cadence. It must not be silently reused as an exact matched
official A result. No new Houston training is launched in this HyRANK batch.

Pavia and all historical experiment files are untouched.
