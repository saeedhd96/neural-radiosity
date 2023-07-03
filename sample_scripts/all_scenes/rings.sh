bash sample_scripts/all_scenes/run_nerad.sh \
dataset.scene=data/NeRad_paper_scenes/rings/scene.xml \
name=rings \
rendering=nerad_specular \
rendering.spp=512 \
${@}
