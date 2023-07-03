bash sample_scripts/all_scenes/run_nerad.sh \
dataset.scene=data/NeRad_paper_scenes/chair/scene_mirror.xml \
name=chair_mirror \
rendering=nerad_specular \
rendering.spp=512 \
${@}
