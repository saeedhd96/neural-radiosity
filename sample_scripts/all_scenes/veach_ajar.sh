bash sample_scripts/all_scenes/run_nerad.sh \
dataset.scene=data/NeRad_paper_scenes/veach_ajar/scene.xml \
name=veach_ajar \
n_steps=120000 \
rendering=nerad \
learning_rate=0.00005 \
lr_decay_start=20000 \
lr_decay_rate=0.35 \
lr_decay_steps=20000 \
lr_decay_min_rate=0.01 \
validation.image.step_size=20000 \
clipgrads=20000 \
${@}
