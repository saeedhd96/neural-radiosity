bash sample_scripts/all_scenes/run_nerad.sh \
dataset.scene=data/NeRad_paper_scenes/dining_room/scene.xml \
name=dining_room \
rendering/pos_emb=dense_grid \
rendering.pos_emb.resolution=512 \
${@}
