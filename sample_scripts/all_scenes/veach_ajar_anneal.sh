bash sample_scripts/all_scenes/veach_ajar.sh \
name=veach_ajar_relative_both_anneal \
residual_loss=l2_relative_both \
residual_loss.config.annealing=true \
${@}
