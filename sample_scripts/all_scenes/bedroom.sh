bash sample_scripts/all_scenes/run_nerad.sh \
dataset.scene=data/NeRad_paper_scenes/bedroom_orig/scene.xml \
name=bedroom \
n_steps=100000 \
rendering=nerad_specular \
lr_decay_start=30000 \
lr_decay_rate=0.35 \
lr_decay_steps=30000 \
lr_decay_min_rate=0.01 \
validation.image.step_size=10000 \
${@}
