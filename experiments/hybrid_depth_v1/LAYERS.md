# Hybrid depth V1: residual-stage probes

Fixed2048 source/target samples identical to the previous probe. Eval mode,
no backward/updates. Hook logits were verified unchanged; state hashes unchanged.
Cosines are to each model's local final classifier w6; early stages are not
in the final LayerNorm coordinate system. The auxiliary margin applies the
frozen final norm/head to early vectors; it is not a trained early-exit head.
Initial CLS has concentration1 by construction: this is not learned collapse.

## Seed2101 epoch200 stage comparison

| Arm | Domain | Stage | Norm | Direction concentration | Raw cos(w6) | Auxiliary margin6 | Auxiliary class6 share |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bida3 | source | tokenizer_all | 4.290 | 0.730 | -0.041 | -1.610 | 31.65% |
| bida3 | source | tokenizer_mean | 4.279 | 0.731 | -0.041 | -1.610 | 31.25% |
| bida3 | source | block1_input_cls | 0.290 | 1.000 | -0.002 | -1.963 | 0.00% |
| bida3 | source | block1_attention_update | 2.455 | 0.293 | -0.002 | -3.639 | 15.38% |
| bida3 | source | block1_after_attention | 2.466 | 0.306 | -0.001 | -3.689 | 15.33% |
| bida3 | source | block1_mlp_update | 1.803 | 0.352 | 0.050 | -3.699 | 15.92% |
| bida3 | source | block1_after_mlp | 3.174 | 0.266 | 0.026 | -4.983 | 15.92% |
| bida3 | source | block2_input_cls | 3.174 | 0.266 | 0.026 | -4.983 | 15.92% |
| bida3 | source | block2_attention_update | 2.776 | 0.259 | 0.031 | -4.635 | 15.87% |
| bida3 | source | block2_after_attention | 4.701 | 0.198 | 0.033 | -6.110 | 15.92% |
| bida3 | source | block2_mlp_update | 1.750 | 0.371 | 0.039 | -3.315 | 15.92% |
| bida3 | source | block2_after_mlp | 5.309 | 0.173 | 0.044 | -6.522 | 15.92% |
| bida3 | source | block3_input_cls | 5.309 | 0.173 | 0.044 | -6.522 | 15.92% |
| bida3 | source | block3_attention_update | 2.600 | 0.235 | 0.068 | -3.493 | 15.92% |
| bida3 | source | block3_after_attention | 6.450 | 0.151 | 0.063 | -6.678 | 15.92% |
| bida3 | source | block3_mlp_update | 1.682 | 0.371 | 0.086 | -3.039 | 16.46% |
| bida3 | source | block3_after_mlp | 6.982 | 0.141 | 0.078 | -6.895 | 15.92% |
| bida3 | source | final_cls | 8.139 | 0.142 | 0.080 | -6.895 | 15.92% |
| bida3 | target | tokenizer_all | 4.403 | 0.851 | -0.066 | -2.563 | 7.64% |
| bida3 | target | tokenizer_mean | 4.385 | 0.854 | -0.066 | -2.564 | 7.18% |
| bida3 | target | block1_input_cls | 0.290 | 1.000 | -0.002 | -1.963 | 0.00% |
| bida3 | target | block1_attention_update | 2.477 | 0.571 | 0.198 | 0.343 | 62.55% |
| bida3 | target | block1_after_attention | 2.479 | 0.573 | 0.198 | 0.317 | 62.06% |
| bida3 | target | block1_mlp_update | 1.928 | 0.634 | 0.204 | 0.525 | 60.06% |
| bida3 | target | block1_after_mlp | 3.428 | 0.596 | 0.255 | 0.933 | 61.57% |
| bida3 | target | block2_input_cls | 3.428 | 0.596 | 0.255 | 0.933 | 61.57% |
| bida3 | target | block2_attention_update | 2.995 | 0.659 | 0.237 | 0.755 | 58.98% |
| bida3 | target | block2_after_attention | 4.971 | 0.610 | 0.310 | 1.168 | 60.21% |
| bida3 | target | block2_mlp_update | 1.744 | 0.599 | 0.208 | 0.659 | 54.83% |
| bida3 | target | block2_after_mlp | 5.579 | 0.617 | 0.340 | 1.283 | 58.98% |
| bida3 | target | block3_input_cls | 5.579 | 0.617 | 0.340 | 1.283 | 58.98% |
| bida3 | target | block3_attention_update | 2.518 | 0.638 | 0.298 | 0.652 | 62.40% |
| bida3 | target | block3_after_attention | 6.591 | 0.610 | 0.398 | 1.742 | 60.45% |
| bida3 | target | block3_mlp_update | 1.760 | 0.717 | 0.298 | 1.349 | 71.00% |
| bida3 | target | block3_after_mlp | 7.083 | 0.618 | 0.443 | 2.184 | 62.21% |
| bida3 | target | final_cls | 8.151 | 0.618 | 0.447 | 2.184 | 62.21% |
| query1 | source | tokenizer_all | 2.374 | 0.322 | -0.017 | -2.382 | 10.52% |
| query1 | source | tokenizer_mean | 2.356 | 0.324 | -0.018 | -2.371 | 11.13% |
| query1 | source | block1_input_cls | 0.274 | 1.000 | -0.004 | -3.159 | 0.00% |
| query1 | source | block1_attention_update | 2.783 | 0.170 | 0.019 | -5.947 | 15.92% |
| query1 | source | block1_after_attention | 2.789 | 0.181 | 0.021 | -5.941 | 15.92% |
| query1 | source | block1_mlp_update | 1.921 | 0.255 | 0.010 | -5.904 | 15.92% |
| query1 | source | block1_after_mlp | 3.728 | 0.115 | 0.018 | -7.511 | 15.92% |
| query1 | source | final_cls | 8.117 | 0.107 | 0.020 | -7.511 | 15.92% |
| query1 | target | tokenizer_all | 2.937 | 0.852 | -0.057 | -2.695 | 1.14% |
| query1 | target | tokenizer_mean | 2.930 | 0.854 | -0.057 | -2.682 | 0.93% |
| query1 | target | block1_input_cls | 0.274 | 1.000 | -0.004 | -3.159 | 0.00% |
| query1 | target | block1_attention_update | 2.804 | 0.889 | 0.414 | 2.565 | 79.74% |
| query1 | target | block1_after_attention | 2.804 | 0.889 | 0.415 | 2.520 | 79.74% |
| query1 | target | block1_mlp_update | 1.924 | 0.897 | 0.507 | 3.894 | 85.94% |
| query1 | target | block1_after_mlp | 3.587 | 0.882 | 0.594 | 4.374 | 83.25% |
| query1 | target | final_cls | 8.114 | 0.882 | 0.595 | 4.374 | 83.25% |
| query3 | source | tokenizer_all | 2.480 | 0.310 | -0.018 | -1.926 | 19.56% |
| query3 | source | tokenizer_mean | 2.468 | 0.312 | -0.018 | -1.929 | 19.73% |
| query3 | source | block1_input_cls | 0.301 | 1.000 | 0.143 | -2.392 | 0.00% |
| query3 | source | block1_attention_update | 2.543 | 0.243 | 0.015 | -3.838 | 15.53% |
| query3 | source | block1_after_attention | 2.565 | 0.274 | 0.032 | -3.666 | 15.67% |
| query3 | source | block1_mlp_update | 1.717 | 0.365 | 0.002 | -4.162 | 15.92% |
| query3 | source | block1_after_mlp | 3.247 | 0.240 | 0.023 | -5.024 | 15.92% |
| query3 | source | block2_input_cls | 3.247 | 0.240 | 0.023 | -5.024 | 15.92% |
| query3 | source | block2_attention_update | 2.775 | 0.235 | -0.038 | -5.214 | 15.92% |
| query3 | source | block2_after_attention | 4.788 | 0.198 | 0.002 | -6.262 | 15.92% |
| query3 | source | block2_mlp_update | 1.665 | 0.380 | 0.044 | -2.782 | 15.92% |
| query3 | source | block2_after_mlp | 5.312 | 0.178 | 0.011 | -6.518 | 15.92% |
| query3 | source | block3_input_cls | 5.312 | 0.178 | 0.011 | -6.518 | 15.92% |
| query3 | source | block3_attention_update | 2.651 | 0.239 | 0.081 | -3.080 | 15.92% |
| query3 | source | block3_after_attention | 6.291 | 0.142 | 0.038 | -6.804 | 15.92% |
| query3 | source | block3_mlp_update | 1.692 | 0.373 | 0.065 | -3.300 | 13.77% |
| query3 | source | block3_after_mlp | 6.772 | 0.144 | 0.056 | -6.997 | 15.92% |
| query3 | source | final_cls | 8.108 | 0.143 | 0.057 | -6.997 | 15.92% |
| query3 | target | tokenizer_all | 2.847 | 0.868 | -0.080 | -1.601 | 5.96% |
| query3 | target | tokenizer_mean | 2.839 | 0.870 | -0.080 | -1.592 | 6.15% |
| query3 | target | block1_input_cls | 0.301 | 1.000 | 0.143 | -2.392 | 0.00% |
| query3 | target | block1_attention_update | 2.654 | 0.906 | 0.348 | 2.146 | 96.24% |
| query3 | target | block1_after_attention | 2.689 | 0.908 | 0.360 | 2.172 | 95.61% |
| query3 | target | block1_mlp_update | 1.762 | 0.920 | 0.382 | 2.884 | 97.61% |
| query3 | target | block1_after_mlp | 3.418 | 0.922 | 0.480 | 3.646 | 98.93% |
| query3 | target | block2_input_cls | 3.418 | 0.922 | 0.480 | 3.646 | 98.93% |
| query3 | target | block2_attention_update | 2.941 | 0.899 | 0.380 | 3.865 | 98.49% |
| query3 | target | block2_after_attention | 4.849 | 0.911 | 0.568 | 5.601 | 98.97% |
| query3 | target | block2_mlp_update | 1.795 | 0.923 | 0.454 | 3.799 | 95.51% |
| query3 | target | block2_after_mlp | 5.520 | 0.920 | 0.646 | 6.683 | 99.46% |
| query3 | target | block3_input_cls | 5.520 | 0.920 | 0.646 | 6.683 | 99.46% |
| query3 | target | block3_attention_update | 2.367 | 0.903 | 0.371 | 1.773 | 85.50% |
| query3 | target | block3_after_attention | 6.443 | 0.924 | 0.688 | 6.927 | 98.14% |
| query3 | target | block3_mlp_update | 1.567 | 0.922 | 0.133 | -0.350 | 32.81% |
| query3 | target | block3_after_mlp | 6.729 | 0.920 | 0.690 | 6.754 | 97.56% |
| query3 | target | final_cls | 8.109 | 0.920 | 0.693 | 6.754 | 97.56% |

## Seed2101 target trajectory by residual stage

| Arm | Epoch | Stage | Concentration | Raw cos(w6) | Auxiliary margin6 |
| --- | ---: | --- | ---: | ---: | ---: |
| bida3 | 10 | tokenizer_mean | 0.868 | -0.111 | -3.235 |
| bida3 | 10 | block1_after_attention | 0.707 | 0.253 | 1.785 |
| bida3 | 10 | block1_after_mlp | 0.737 | 0.300 | 2.163 |
| bida3 | 10 | block2_after_attention | 0.742 | 0.349 | 2.306 |
| bida3 | 10 | block2_after_mlp | 0.755 | 0.397 | 2.909 |
| bida3 | 10 | block3_after_attention | 0.758 | 0.453 | 3.240 |
| bida3 | 10 | block3_after_mlp | 0.760 | 0.496 | 3.624 |
| bida3 | 10 | final_cls | 0.761 | 0.499 | 3.624 |
| bida3 | 20 | tokenizer_mean | 0.856 | -0.060 | -2.465 |
| bida3 | 20 | block1_after_attention | 0.549 | 0.191 | 0.321 |
| bida3 | 20 | block1_after_mlp | 0.551 | 0.216 | 0.672 |
| bida3 | 20 | block2_after_attention | 0.548 | 0.239 | 0.327 |
| bida3 | 20 | block2_after_mlp | 0.554 | 0.249 | 0.306 |
| bida3 | 20 | block3_after_attention | 0.528 | 0.265 | 0.052 |
| bida3 | 20 | block3_after_mlp | 0.528 | 0.301 | 0.416 |
| bida3 | 20 | final_cls | 0.527 | 0.305 | 0.416 |
| bida3 | 30 | tokenizer_mean | 0.859 | -0.079 | -2.594 |
| bida3 | 30 | block1_after_attention | 0.658 | 0.227 | 0.954 |
| bida3 | 30 | block1_after_mlp | 0.671 | 0.263 | 1.482 |
| bida3 | 30 | block2_after_attention | 0.673 | 0.316 | 1.684 |
| bida3 | 30 | block2_after_mlp | 0.680 | 0.349 | 1.915 |
| bida3 | 30 | block3_after_attention | 0.680 | 0.409 | 2.216 |
| bida3 | 30 | block3_after_mlp | 0.686 | 0.460 | 2.774 |
| bida3 | 30 | final_cls | 0.687 | 0.463 | 2.774 |
| bida3 | 40 | tokenizer_mean | 0.848 | -0.061 | -2.310 |
| bida3 | 40 | block1_after_attention | 0.599 | 0.228 | 0.911 |
| bida3 | 40 | block1_after_mlp | 0.621 | 0.266 | 1.251 |
| bida3 | 40 | block2_after_attention | 0.621 | 0.310 | 1.294 |
| bida3 | 40 | block2_after_mlp | 0.629 | 0.344 | 1.497 |
| bida3 | 40 | block3_after_attention | 0.626 | 0.401 | 1.851 |
| bida3 | 40 | block3_after_mlp | 0.632 | 0.443 | 2.255 |
| bida3 | 40 | final_cls | 0.634 | 0.446 | 2.255 |
| bida3 | 50 | tokenizer_mean | 0.852 | -0.097 | -3.018 |
| bida3 | 50 | block1_after_attention | 0.628 | 0.192 | 0.604 |
| bida3 | 50 | block1_after_mlp | 0.639 | 0.233 | 1.041 |
| bida3 | 50 | block2_after_attention | 0.646 | 0.295 | 1.327 |
| bida3 | 50 | block2_after_mlp | 0.653 | 0.323 | 1.449 |
| bida3 | 50 | block3_after_attention | 0.648 | 0.363 | 1.655 |
| bida3 | 50 | block3_after_mlp | 0.650 | 0.407 | 2.083 |
| bida3 | 50 | final_cls | 0.651 | 0.410 | 2.083 |
| bida3 | 60 | tokenizer_mean | 0.864 | -0.090 | -2.788 |
| bida3 | 60 | block1_after_attention | 0.646 | 0.231 | 1.220 |
| bida3 | 60 | block1_after_mlp | 0.652 | 0.261 | 1.449 |
| bida3 | 60 | block2_after_attention | 0.649 | 0.303 | 1.425 |
| bida3 | 60 | block2_after_mlp | 0.655 | 0.328 | 1.521 |
| bida3 | 60 | block3_after_attention | 0.650 | 0.376 | 1.708 |
| bida3 | 60 | block3_after_mlp | 0.655 | 0.425 | 2.200 |
| bida3 | 60 | final_cls | 0.656 | 0.428 | 2.200 |
| bida3 | 70 | tokenizer_mean | 0.858 | -0.076 | -2.464 |
| bida3 | 70 | block1_after_attention | 0.569 | 0.180 | 0.368 |
| bida3 | 70 | block1_after_mlp | 0.576 | 0.211 | 0.579 |
| bida3 | 70 | block2_after_attention | 0.580 | 0.254 | 0.542 |
| bida3 | 70 | block2_after_mlp | 0.585 | 0.278 | 0.593 |
| bida3 | 70 | block3_after_attention | 0.578 | 0.313 | 0.674 |
| bida3 | 70 | block3_after_mlp | 0.582 | 0.358 | 1.070 |
| bida3 | 70 | final_cls | 0.583 | 0.361 | 1.070 |
| bida3 | 80 | tokenizer_mean | 0.858 | -0.074 | -2.452 |
| bida3 | 80 | block1_after_attention | 0.577 | 0.193 | 0.701 |
| bida3 | 80 | block1_after_mlp | 0.583 | 0.224 | 0.750 |
| bida3 | 80 | block2_after_attention | 0.590 | 0.267 | 0.678 |
| bida3 | 80 | block2_after_mlp | 0.594 | 0.291 | 0.734 |
| bida3 | 80 | block3_after_attention | 0.591 | 0.339 | 0.938 |
| bida3 | 80 | block3_after_mlp | 0.600 | 0.388 | 1.398 |
| bida3 | 80 | final_cls | 0.601 | 0.390 | 1.398 |
| bida3 | 90 | tokenizer_mean | 0.861 | -0.072 | -2.558 |
| bida3 | 90 | block1_after_attention | 0.633 | 0.230 | 1.172 |
| bida3 | 90 | block1_after_mlp | 0.639 | 0.264 | 1.389 |
| bida3 | 90 | block2_after_attention | 0.634 | 0.308 | 1.362 |
| bida3 | 90 | block2_after_mlp | 0.642 | 0.339 | 1.560 |
| bida3 | 90 | block3_after_attention | 0.642 | 0.389 | 1.729 |
| bida3 | 90 | block3_after_mlp | 0.646 | 0.437 | 2.178 |
| bida3 | 90 | final_cls | 0.647 | 0.440 | 2.178 |
| bida3 | 100 | tokenizer_mean | 0.866 | -0.084 | -2.873 |
| bida3 | 100 | block1_after_attention | 0.651 | 0.232 | 1.170 |
| bida3 | 100 | block1_after_mlp | 0.657 | 0.267 | 1.415 |
| bida3 | 100 | block2_after_attention | 0.659 | 0.328 | 1.613 |
| bida3 | 100 | block2_after_mlp | 0.667 | 0.357 | 1.762 |
| bida3 | 100 | block3_after_attention | 0.660 | 0.404 | 1.929 |
| bida3 | 100 | block3_after_mlp | 0.663 | 0.450 | 2.351 |
| bida3 | 100 | final_cls | 0.664 | 0.453 | 2.351 |
| bida3 | 110 | tokenizer_mean | 0.869 | -0.089 | -2.877 |
| bida3 | 110 | block1_after_attention | 0.650 | 0.232 | 1.088 |
| bida3 | 110 | block1_after_mlp | 0.660 | 0.275 | 1.478 |
| bida3 | 110 | block2_after_attention | 0.665 | 0.330 | 1.691 |
| bida3 | 110 | block2_after_mlp | 0.673 | 0.365 | 1.948 |
| bida3 | 110 | block3_after_attention | 0.671 | 0.419 | 2.183 |
| bida3 | 110 | block3_after_mlp | 0.676 | 0.466 | 2.620 |
| bida3 | 110 | final_cls | 0.677 | 0.469 | 2.620 |
| bida3 | 120 | tokenizer_mean | 0.863 | -0.085 | -2.846 |
| bida3 | 120 | block1_after_attention | 0.593 | 0.208 | 0.786 |
| bida3 | 120 | block1_after_mlp | 0.597 | 0.242 | 0.895 |
| bida3 | 120 | block2_after_attention | 0.601 | 0.279 | 0.783 |
| bida3 | 120 | block2_after_mlp | 0.606 | 0.307 | 0.899 |
| bida3 | 120 | block3_after_attention | 0.602 | 0.357 | 1.036 |
| bida3 | 120 | block3_after_mlp | 0.609 | 0.402 | 1.473 |
| bida3 | 120 | final_cls | 0.609 | 0.405 | 1.473 |
| bida3 | 130 | tokenizer_mean | 0.864 | -0.080 | -2.787 |
| bida3 | 130 | block1_after_attention | 0.588 | 0.207 | 0.719 |
| bida3 | 130 | block1_after_mlp | 0.593 | 0.241 | 0.820 |
| bida3 | 130 | block2_after_attention | 0.600 | 0.274 | 0.701 |
| bida3 | 130 | block2_after_mlp | 0.604 | 0.300 | 0.774 |
| bida3 | 130 | block3_after_attention | 0.603 | 0.350 | 1.023 |
| bida3 | 130 | block3_after_mlp | 0.609 | 0.397 | 1.452 |
| bida3 | 130 | final_cls | 0.610 | 0.400 | 1.452 |
| bida3 | 140 | tokenizer_mean | 0.860 | -0.077 | -2.659 |
| bida3 | 140 | block1_after_attention | 0.558 | 0.175 | 0.272 |
| bida3 | 140 | block1_after_mlp | 0.562 | 0.208 | 0.288 |
| bida3 | 140 | block2_after_attention | 0.567 | 0.245 | 0.210 |
| bida3 | 140 | block2_after_mlp | 0.572 | 0.271 | 0.280 |
| bida3 | 140 | block3_after_attention | 0.570 | 0.319 | 0.531 |
| bida3 | 140 | block3_after_mlp | 0.575 | 0.363 | 0.918 |
| bida3 | 140 | final_cls | 0.575 | 0.367 | 0.918 |
| bida3 | 150 | tokenizer_mean | 0.850 | -0.064 | -2.261 |
| bida3 | 150 | block1_after_attention | 0.592 | 0.183 | 0.303 |
| bida3 | 150 | block1_after_mlp | 0.603 | 0.231 | 0.758 |
| bida3 | 150 | block2_after_attention | 0.614 | 0.288 | 0.959 |
| bida3 | 150 | block2_after_mlp | 0.621 | 0.323 | 1.149 |
| bida3 | 150 | block3_after_attention | 0.614 | 0.382 | 1.564 |
| bida3 | 150 | block3_after_mlp | 0.618 | 0.431 | 2.082 |
| bida3 | 150 | final_cls | 0.619 | 0.434 | 2.082 |
| bida3 | 160 | tokenizer_mean | 0.857 | -0.050 | -2.081 |
| bida3 | 160 | block1_after_attention | 0.589 | 0.174 | 0.283 |
| bida3 | 160 | block1_after_mlp | 0.597 | 0.215 | 0.530 |
| bida3 | 160 | block2_after_attention | 0.605 | 0.271 | 0.722 |
| bida3 | 160 | block2_after_mlp | 0.612 | 0.304 | 0.913 |
| bida3 | 160 | block3_after_attention | 0.606 | 0.366 | 1.273 |
| bida3 | 160 | block3_after_mlp | 0.610 | 0.415 | 1.789 |
| bida3 | 160 | final_cls | 0.612 | 0.418 | 1.789 |
| bida3 | 170 | tokenizer_mean | 0.860 | -0.054 | -2.147 |
| bida3 | 170 | block1_after_attention | 0.599 | 0.174 | 0.286 |
| bida3 | 170 | block1_after_mlp | 0.605 | 0.213 | 0.494 |
| bida3 | 170 | block2_after_attention | 0.609 | 0.268 | 0.669 |
| bida3 | 170 | block2_after_mlp | 0.615 | 0.300 | 0.837 |
| bida3 | 170 | block3_after_attention | 0.610 | 0.361 | 1.226 |
| bida3 | 170 | block3_after_mlp | 0.613 | 0.409 | 1.717 |
| bida3 | 170 | final_cls | 0.615 | 0.412 | 1.717 |
| bida3 | 180 | tokenizer_mean | 0.859 | -0.051 | -2.119 |
| bida3 | 180 | block1_after_attention | 0.571 | 0.163 | 0.138 |
| bida3 | 180 | block1_after_mlp | 0.576 | 0.201 | 0.300 |
| bida3 | 180 | block2_after_attention | 0.582 | 0.253 | 0.422 |
| bida3 | 180 | block2_after_mlp | 0.586 | 0.282 | 0.550 |
| bida3 | 180 | block3_after_attention | 0.580 | 0.343 | 0.903 |
| bida3 | 180 | block3_after_mlp | 0.584 | 0.389 | 1.370 |
| bida3 | 180 | final_cls | 0.585 | 0.393 | 1.370 |
| bida3 | 190 | tokenizer_mean | 0.858 | -0.051 | -2.078 |
| bida3 | 190 | block1_after_attention | 0.550 | 0.154 | 0.042 |
| bida3 | 190 | block1_after_mlp | 0.551 | 0.186 | 0.038 |
| bida3 | 190 | block2_after_attention | 0.555 | 0.228 | 0.002 |
| bida3 | 190 | block2_after_mlp | 0.558 | 0.254 | 0.077 |
| bida3 | 190 | block3_after_attention | 0.551 | 0.309 | 0.345 |
| bida3 | 190 | block3_after_mlp | 0.554 | 0.352 | 0.759 |
| bida3 | 190 | final_cls | 0.555 | 0.356 | 0.759 |
| bida3 | 200 | tokenizer_mean | 0.854 | -0.066 | -2.564 |
| bida3 | 200 | block1_after_attention | 0.573 | 0.198 | 0.317 |
| bida3 | 200 | block1_after_mlp | 0.596 | 0.255 | 0.933 |
| bida3 | 200 | block2_after_attention | 0.610 | 0.310 | 1.168 |
| bida3 | 200 | block2_after_mlp | 0.617 | 0.340 | 1.283 |
| bida3 | 200 | block3_after_attention | 0.610 | 0.398 | 1.742 |
| bida3 | 200 | block3_after_mlp | 0.618 | 0.443 | 2.184 |
| bida3 | 200 | final_cls | 0.618 | 0.447 | 2.184 |
| query1 | 10 | tokenizer_mean | 0.841 | -0.042 | -2.592 |
| query1 | 10 | block1_after_attention | 0.855 | 0.275 | 1.111 |
| query1 | 10 | block1_after_mlp | 0.834 | 0.413 | 2.373 |
| query1 | 10 | final_cls | 0.835 | 0.413 | 2.373 |
| query1 | 20 | tokenizer_mean | 0.822 | -0.031 | -2.376 |
| query1 | 20 | block1_after_attention | 0.846 | 0.284 | 1.001 |
| query1 | 20 | block1_after_mlp | 0.823 | 0.430 | 2.285 |
| query1 | 20 | final_cls | 0.824 | 0.430 | 2.285 |
| query1 | 30 | tokenizer_mean | 0.847 | -0.066 | -2.625 |
| query1 | 30 | block1_after_attention | 0.885 | 0.336 | 1.597 |
| query1 | 30 | block1_after_mlp | 0.873 | 0.478 | 3.003 |
| query1 | 30 | final_cls | 0.874 | 0.478 | 3.003 |
| query1 | 40 | tokenizer_mean | 0.851 | -0.044 | -2.409 |
| query1 | 40 | block1_after_attention | 0.885 | 0.340 | 1.618 |
| query1 | 40 | block1_after_mlp | 0.874 | 0.493 | 3.244 |
| query1 | 40 | final_cls | 0.875 | 0.493 | 3.244 |
| query1 | 50 | tokenizer_mean | 0.790 | -0.043 | -2.515 |
| query1 | 50 | block1_after_attention | 0.835 | 0.301 | 1.255 |
| query1 | 50 | block1_after_mlp | 0.810 | 0.444 | 2.445 |
| query1 | 50 | final_cls | 0.811 | 0.445 | 2.445 |
| query1 | 60 | tokenizer_mean | 0.854 | -0.059 | -2.663 |
| query1 | 60 | block1_after_attention | 0.891 | 0.379 | 2.352 |
| query1 | 60 | block1_after_mlp | 0.883 | 0.537 | 3.820 |
| query1 | 60 | final_cls | 0.884 | 0.537 | 3.820 |
| query1 | 70 | tokenizer_mean | 0.853 | -0.054 | -2.648 |
| query1 | 70 | block1_after_attention | 0.887 | 0.382 | 2.321 |
| query1 | 70 | block1_after_mlp | 0.879 | 0.539 | 3.780 |
| query1 | 70 | final_cls | 0.880 | 0.540 | 3.780 |
| query1 | 80 | tokenizer_mean | 0.858 | -0.055 | -2.712 |
| query1 | 80 | block1_after_attention | 0.895 | 0.387 | 2.342 |
| query1 | 80 | block1_after_mlp | 0.889 | 0.550 | 3.862 |
| query1 | 80 | final_cls | 0.890 | 0.551 | 3.862 |
| query1 | 90 | tokenizer_mean | 0.860 | -0.055 | -2.707 |
| query1 | 90 | block1_after_attention | 0.897 | 0.388 | 2.326 |
| query1 | 90 | block1_after_mlp | 0.890 | 0.555 | 3.918 |
| query1 | 90 | final_cls | 0.891 | 0.556 | 3.918 |
| query1 | 100 | tokenizer_mean | 0.862 | -0.052 | -2.661 |
| query1 | 100 | block1_after_attention | 0.898 | 0.392 | 2.376 |
| query1 | 100 | block1_after_mlp | 0.891 | 0.562 | 4.050 |
| query1 | 100 | final_cls | 0.891 | 0.562 | 4.050 |
| query1 | 110 | tokenizer_mean | 0.863 | -0.051 | -2.658 |
| query1 | 110 | block1_after_attention | 0.900 | 0.393 | 2.383 |
| query1 | 110 | block1_after_mlp | 0.892 | 0.565 | 4.086 |
| query1 | 110 | final_cls | 0.893 | 0.566 | 4.086 |
| query1 | 120 | tokenizer_mean | 0.845 | -0.061 | -2.720 |
| query1 | 120 | block1_after_attention | 0.888 | 0.397 | 2.354 |
| query1 | 120 | block1_after_mlp | 0.880 | 0.559 | 3.778 |
| query1 | 120 | final_cls | 0.880 | 0.560 | 3.778 |
| query1 | 130 | tokenizer_mean | 0.845 | -0.058 | -2.716 |
| query1 | 130 | block1_after_attention | 0.883 | 0.399 | 2.375 |
| query1 | 130 | block1_after_mlp | 0.875 | 0.562 | 3.826 |
| query1 | 130 | final_cls | 0.876 | 0.563 | 3.826 |
| query1 | 140 | tokenizer_mean | 0.867 | -0.074 | -2.818 |
| query1 | 140 | block1_after_attention | 0.910 | 0.429 | 2.758 |
| query1 | 140 | block1_after_mlp | 0.906 | 0.607 | 4.396 |
| query1 | 140 | final_cls | 0.906 | 0.608 | 4.396 |
| query1 | 150 | tokenizer_mean | 0.850 | -0.057 | -2.726 |
| query1 | 150 | block1_after_attention | 0.890 | 0.412 | 2.519 |
| query1 | 150 | block1_after_mlp | 0.883 | 0.584 | 4.144 |
| query1 | 150 | final_cls | 0.884 | 0.585 | 4.144 |
| query1 | 160 | tokenizer_mean | 0.853 | -0.059 | -2.705 |
| query1 | 160 | block1_after_attention | 0.891 | 0.413 | 2.537 |
| query1 | 160 | block1_after_mlp | 0.885 | 0.587 | 4.216 |
| query1 | 160 | final_cls | 0.885 | 0.588 | 4.216 |
| query1 | 170 | tokenizer_mean | 0.852 | -0.053 | -2.655 |
| query1 | 170 | block1_after_attention | 0.888 | 0.409 | 2.473 |
| query1 | 170 | block1_after_mlp | 0.881 | 0.584 | 4.228 |
| query1 | 170 | final_cls | 0.881 | 0.585 | 4.228 |
| query1 | 180 | tokenizer_mean | 0.861 | -0.062 | -2.723 |
| query1 | 180 | block1_after_attention | 0.898 | 0.426 | 2.706 |
| query1 | 180 | block1_after_mlp | 0.893 | 0.607 | 4.549 |
| query1 | 180 | final_cls | 0.893 | 0.608 | 4.549 |
| query1 | 190 | tokenizer_mean | 0.852 | -0.053 | -2.642 |
| query1 | 190 | block1_after_attention | 0.887 | 0.411 | 2.465 |
| query1 | 190 | block1_after_mlp | 0.880 | 0.587 | 4.247 |
| query1 | 190 | final_cls | 0.880 | 0.589 | 4.247 |
| query1 | 200 | tokenizer_mean | 0.854 | -0.057 | -2.682 |
| query1 | 200 | block1_after_attention | 0.889 | 0.415 | 2.520 |
| query1 | 200 | block1_after_mlp | 0.882 | 0.594 | 4.374 |
| query1 | 200 | final_cls | 0.882 | 0.595 | 4.374 |
| query3 | 10 | tokenizer_mean | 0.846 | -0.098 | -2.496 |
| query3 | 10 | block1_after_attention | 0.852 | 0.273 | 1.648 |
| query3 | 10 | block1_after_mlp | 0.869 | 0.316 | 2.333 |
| query3 | 10 | block2_after_attention | 0.848 | 0.371 | 3.303 |
| query3 | 10 | block2_after_mlp | 0.860 | 0.418 | 3.832 |
| query3 | 10 | block3_after_attention | 0.863 | 0.465 | 4.138 |
| query3 | 10 | block3_after_mlp | 0.859 | 0.453 | 3.953 |
| query3 | 10 | final_cls | 0.859 | 0.454 | 3.953 |
| query3 | 20 | tokenizer_mean | 0.717 | 0.127 | -0.621 |
| query3 | 20 | block1_after_attention | 0.695 | -0.120 | -2.995 |
| query3 | 20 | block1_after_mlp | 0.647 | -0.109 | -3.365 |
| query3 | 20 | block2_after_attention | 0.614 | -0.103 | -4.272 |
| query3 | 20 | block2_after_mlp | 0.602 | -0.102 | -4.212 |
| query3 | 20 | block3_after_attention | 0.572 | -0.103 | -4.485 |
| query3 | 20 | block3_after_mlp | 0.574 | -0.054 | -4.240 |
| query3 | 20 | final_cls | 0.576 | -0.054 | -4.240 |
| query3 | 30 | tokenizer_mean | 0.860 | -0.105 | -2.251 |
| query3 | 30 | block1_after_attention | 0.899 | 0.326 | 1.908 |
| query3 | 30 | block1_after_mlp | 0.910 | 0.398 | 2.686 |
| query3 | 30 | block2_after_attention | 0.888 | 0.479 | 4.504 |
| query3 | 30 | block2_after_mlp | 0.896 | 0.542 | 5.391 |
| query3 | 30 | block3_after_attention | 0.900 | 0.576 | 5.390 |
| query3 | 30 | block3_after_mlp | 0.898 | 0.564 | 5.176 |
| query3 | 30 | final_cls | 0.899 | 0.566 | 5.176 |
| query3 | 40 | tokenizer_mean | 0.846 | -0.094 | -1.999 |
| query3 | 40 | block1_after_attention | 0.887 | 0.319 | 1.949 |
| query3 | 40 | block1_after_mlp | 0.896 | 0.386 | 2.596 |
| query3 | 40 | block2_after_attention | 0.871 | 0.471 | 4.308 |
| query3 | 40 | block2_after_mlp | 0.875 | 0.525 | 5.007 |
| query3 | 40 | block3_after_attention | 0.876 | 0.561 | 5.087 |
| query3 | 40 | block3_after_mlp | 0.874 | 0.551 | 4.916 |
| query3 | 40 | final_cls | 0.874 | 0.553 | 4.916 |
| query3 | 50 | tokenizer_mean | 0.852 | -0.060 | -1.598 |
| query3 | 50 | block1_after_attention | 0.892 | 0.328 | 1.985 |
| query3 | 50 | block1_after_mlp | 0.906 | 0.411 | 2.968 |
| query3 | 50 | block2_after_attention | 0.891 | 0.496 | 4.817 |
| query3 | 50 | block2_after_mlp | 0.897 | 0.562 | 5.683 |
| query3 | 50 | block3_after_attention | 0.900 | 0.588 | 5.608 |
| query3 | 50 | block3_after_mlp | 0.899 | 0.583 | 5.445 |
| query3 | 50 | final_cls | 0.899 | 0.585 | 5.445 |
| query3 | 60 | tokenizer_mean | 0.878 | -0.077 | -1.731 |
| query3 | 60 | block1_after_attention | 0.911 | 0.350 | 2.169 |
| query3 | 60 | block1_after_mlp | 0.924 | 0.441 | 3.154 |
| query3 | 60 | block2_after_attention | 0.911 | 0.531 | 5.201 |
| query3 | 60 | block2_after_mlp | 0.917 | 0.603 | 6.188 |
| query3 | 60 | block3_after_attention | 0.921 | 0.639 | 6.271 |
| query3 | 60 | block3_after_mlp | 0.919 | 0.632 | 6.041 |
| query3 | 60 | final_cls | 0.919 | 0.634 | 6.041 |
| query3 | 70 | tokenizer_mean | 0.877 | -0.074 | -1.743 |
| query3 | 70 | block1_after_attention | 0.906 | 0.353 | 2.175 |
| query3 | 70 | block1_after_mlp | 0.919 | 0.445 | 3.181 |
| query3 | 70 | block2_after_attention | 0.908 | 0.537 | 5.224 |
| query3 | 70 | block2_after_mlp | 0.914 | 0.611 | 6.250 |
| query3 | 70 | block3_after_attention | 0.918 | 0.648 | 6.336 |
| query3 | 70 | block3_after_mlp | 0.916 | 0.642 | 6.115 |
| query3 | 70 | final_cls | 0.916 | 0.644 | 6.115 |
| query3 | 80 | tokenizer_mean | 0.877 | -0.069 | -1.680 |
| query3 | 80 | block1_after_attention | 0.905 | 0.354 | 2.167 |
| query3 | 80 | block1_after_mlp | 0.918 | 0.447 | 3.199 |
| query3 | 80 | block2_after_attention | 0.906 | 0.538 | 5.238 |
| query3 | 80 | block2_after_mlp | 0.913 | 0.614 | 6.289 |
| query3 | 80 | block3_after_attention | 0.918 | 0.650 | 6.344 |
| query3 | 80 | block3_after_mlp | 0.915 | 0.646 | 6.132 |
| query3 | 80 | final_cls | 0.915 | 0.648 | 6.132 |
| query3 | 90 | tokenizer_mean | 0.878 | -0.066 | -1.657 |
| query3 | 90 | block1_after_attention | 0.906 | 0.355 | 2.178 |
| query3 | 90 | block1_after_mlp | 0.919 | 0.454 | 3.314 |
| query3 | 90 | block2_after_attention | 0.909 | 0.547 | 5.361 |
| query3 | 90 | block2_after_mlp | 0.915 | 0.623 | 6.411 |
| query3 | 90 | block3_after_attention | 0.920 | 0.661 | 6.495 |
| query3 | 90 | block3_after_mlp | 0.917 | 0.657 | 6.293 |
| query3 | 90 | final_cls | 0.917 | 0.659 | 6.293 |
| query3 | 100 | tokenizer_mean | 0.878 | -0.064 | -1.629 |
| query3 | 100 | block1_after_attention | 0.906 | 0.356 | 2.188 |
| query3 | 100 | block1_after_mlp | 0.918 | 0.459 | 3.390 |
| query3 | 100 | block2_after_attention | 0.906 | 0.551 | 5.364 |
| query3 | 100 | block2_after_mlp | 0.913 | 0.627 | 6.427 |
| query3 | 100 | block3_after_attention | 0.917 | 0.666 | 6.486 |
| query3 | 100 | block3_after_mlp | 0.915 | 0.663 | 6.317 |
| query3 | 100 | final_cls | 0.915 | 0.664 | 6.317 |
| query3 | 110 | tokenizer_mean | 0.884 | -0.063 | -1.608 |
| query3 | 110 | block1_after_attention | 0.911 | 0.360 | 2.250 |
| query3 | 110 | block1_after_mlp | 0.923 | 0.465 | 3.536 |
| query3 | 110 | block2_after_attention | 0.912 | 0.558 | 5.505 |
| query3 | 110 | block2_after_mlp | 0.919 | 0.635 | 6.592 |
| query3 | 110 | block3_after_attention | 0.923 | 0.675 | 6.646 |
| query3 | 110 | block3_after_mlp | 0.920 | 0.673 | 6.503 |
| query3 | 110 | final_cls | 0.921 | 0.674 | 6.503 |
| query3 | 120 | tokenizer_mean | 0.881 | -0.073 | -1.618 |
| query3 | 120 | block1_after_attention | 0.909 | 0.366 | 2.304 |
| query3 | 120 | block1_after_mlp | 0.921 | 0.473 | 3.574 |
| query3 | 120 | block2_after_attention | 0.909 | 0.562 | 5.432 |
| query3 | 120 | block2_after_mlp | 0.916 | 0.637 | 6.566 |
| query3 | 120 | block3_after_attention | 0.919 | 0.678 | 6.563 |
| query3 | 120 | block3_after_mlp | 0.916 | 0.675 | 6.428 |
| query3 | 120 | final_cls | 0.917 | 0.677 | 6.428 |
| query3 | 130 | tokenizer_mean | 0.883 | -0.075 | -1.651 |
| query3 | 130 | block1_after_attention | 0.910 | 0.365 | 2.328 |
| query3 | 130 | block1_after_mlp | 0.922 | 0.472 | 3.541 |
| query3 | 130 | block2_after_attention | 0.911 | 0.562 | 5.440 |
| query3 | 130 | block2_after_mlp | 0.918 | 0.637 | 6.555 |
| query3 | 130 | block3_after_attention | 0.921 | 0.677 | 6.589 |
| query3 | 130 | block3_after_mlp | 0.919 | 0.673 | 6.408 |
| query3 | 130 | final_cls | 0.919 | 0.675 | 6.408 |
| query3 | 140 | tokenizer_mean | 0.881 | -0.064 | -1.591 |
| query3 | 140 | block1_after_attention | 0.907 | 0.357 | 2.210 |
| query3 | 140 | block1_after_mlp | 0.918 | 0.468 | 3.564 |
| query3 | 140 | block2_after_attention | 0.908 | 0.558 | 5.482 |
| query3 | 140 | block2_after_mlp | 0.915 | 0.634 | 6.568 |
| query3 | 140 | block3_after_attention | 0.919 | 0.672 | 6.562 |
| query3 | 140 | block3_after_mlp | 0.916 | 0.669 | 6.413 |
| query3 | 140 | final_cls | 0.916 | 0.671 | 6.413 |
| query3 | 150 | tokenizer_mean | 0.886 | -0.068 | -1.599 |
| query3 | 150 | block1_after_attention | 0.912 | 0.365 | 2.316 |
| query3 | 150 | block1_after_mlp | 0.924 | 0.478 | 3.687 |
| query3 | 150 | block2_after_attention | 0.914 | 0.568 | 5.633 |
| query3 | 150 | block2_after_mlp | 0.921 | 0.644 | 6.728 |
| query3 | 150 | block3_after_attention | 0.924 | 0.685 | 6.773 |
| query3 | 150 | block3_after_mlp | 0.921 | 0.682 | 6.627 |
| query3 | 150 | final_cls | 0.921 | 0.684 | 6.627 |
| query3 | 160 | tokenizer_mean | 0.844 | -0.099 | -1.769 |
| query3 | 160 | block1_after_attention | 0.878 | 0.342 | 1.916 |
| query3 | 160 | block1_after_mlp | 0.890 | 0.447 | 2.927 |
| query3 | 160 | block2_after_attention | 0.870 | 0.531 | 4.721 |
| query3 | 160 | block2_after_mlp | 0.877 | 0.604 | 5.656 |
| query3 | 160 | block3_after_attention | 0.881 | 0.650 | 6.038 |
| query3 | 160 | block3_after_mlp | 0.875 | 0.649 | 5.766 |
| query3 | 160 | final_cls | 0.876 | 0.651 | 5.766 |
| query3 | 170 | tokenizer_mean | 0.865 | -0.081 | -1.580 |
| query3 | 170 | block1_after_attention | 0.905 | 0.348 | 2.040 |
| query3 | 170 | block1_after_mlp | 0.918 | 0.463 | 3.402 |
| query3 | 170 | block2_after_attention | 0.905 | 0.554 | 5.401 |
| query3 | 170 | block2_after_mlp | 0.914 | 0.633 | 6.441 |
| query3 | 170 | block3_after_attention | 0.918 | 0.672 | 6.699 |
| query3 | 170 | block3_after_mlp | 0.914 | 0.674 | 6.508 |
| query3 | 170 | final_cls | 0.915 | 0.677 | 6.508 |
| query3 | 180 | tokenizer_mean | 0.864 | -0.082 | -1.613 |
| query3 | 180 | block1_after_attention | 0.906 | 0.354 | 2.082 |
| query3 | 180 | block1_after_mlp | 0.920 | 0.470 | 3.458 |
| query3 | 180 | block2_after_attention | 0.907 | 0.560 | 5.414 |
| query3 | 180 | block2_after_mlp | 0.916 | 0.638 | 6.497 |
| query3 | 180 | block3_after_attention | 0.920 | 0.679 | 6.733 |
| query3 | 180 | block3_after_mlp | 0.916 | 0.680 | 6.552 |
| query3 | 180 | final_cls | 0.917 | 0.683 | 6.552 |
| query3 | 190 | tokenizer_mean | 0.872 | -0.083 | -1.590 |
| query3 | 190 | block1_after_attention | 0.912 | 0.363 | 2.209 |
| query3 | 190 | block1_after_mlp | 0.925 | 0.480 | 3.607 |
| query3 | 190 | block2_after_attention | 0.914 | 0.568 | 5.576 |
| query3 | 190 | block2_after_mlp | 0.922 | 0.646 | 6.673 |
| query3 | 190 | block3_after_attention | 0.925 | 0.688 | 6.904 |
| query3 | 190 | block3_after_mlp | 0.922 | 0.689 | 6.737 |
| query3 | 190 | final_cls | 0.923 | 0.692 | 6.737 |
| query3 | 200 | tokenizer_mean | 0.870 | -0.080 | -1.592 |
| query3 | 200 | block1_after_attention | 0.908 | 0.360 | 2.172 |
| query3 | 200 | block1_after_mlp | 0.922 | 0.480 | 3.646 |
| query3 | 200 | block2_after_attention | 0.911 | 0.568 | 5.601 |
| query3 | 200 | block2_after_mlp | 0.920 | 0.646 | 6.683 |
| query3 | 200 | block3_after_attention | 0.924 | 0.688 | 6.927 |
| query3 | 200 | block3_after_mlp | 0.920 | 0.690 | 6.754 |
| query3 | 200 | final_cls | 0.920 | 0.693 | 6.754 |

## Interpretation of the seed2101 boundary probes

Depth3 epoch200: the token-mean concentration is similar for Query and BiDA (0.870 versus 0.854), but block1 after-attention CLS differs sharply (0.908 versus 0.573). After block1 MLP the Query concentration is already 0.922, compared with BiDA 0.596. The pronounced CLS concentration is therefore present in block1, not first appearing in block3.
For the depth3 Query model, raw local cosine to w6 is 0.360 after block1 attention, 0.480 after block1 MLP, 0.568 after block2 attention, 0.646 after block2 MLP, 0.688 after block3 attention, and 0.690 after block3 MLP. Block2 further strengthens alignment; there is no unique single guilty sublayer.
The independently trained depth1 model also shows high target concentration: block1 after attention 0.889 and after MLP 0.882. MLP slightly reduces concentration here while increasing cosine to w6 from 0.415 to 0.594. Concentration and class-specific alignment must be distinguished; not every MLP monotonically concentrates directions.
Source depth1 final concentration is only 0.107, compared with target 0.882, so the signature is domain-specific. Final target concentration is already high at epoch10 (0.835), rises to 0.884 by epoch60, and is 0.882 at epoch200. Depth3 final target concentration is 0.920 at epoch200.
These comparisons show residual-stage signatures in the learned system, not proof that changing an attention/MLP component will repair it. Token-mean and CLS are distinct vectors; initial CLS is constant. Direct cosine to the final head is a descriptive coordinate lens, not a validated early-exit predictor.

These are observational probes, not causal layer lesions. Depth1 retraining
changes all learned weights and BN statistics over optimization. Attribution
to a particular attention or MLP component requires a separate intervention.
