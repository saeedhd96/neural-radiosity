# Uses dense grid encoding
OUT=output/nerad_dense_grid
SCENES=sample_scripts/all_scenes

bash $SCENES/cbox.sh                  out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=64
bash $SCENES/copperman.sh             out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=64
bash $SCENES/bedroom.sh               out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=64
bash $SCENES/veach_ajar.sh            out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=128   #artifacts
bash $SCENES/veach_ajar_mit.sh        out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=64
bash $SCENES/dining_room.sh           out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=128
bash $SCENES/living_room.sh           out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=64
bash $SCENES/chair.sh                 out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=64
bash $SCENES/chair_mirror.sh          out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=64
bash $SCENES/rings.sh                 out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=128
bash $SCENES/veach_ajar_anneal.sh     out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=128   #The loss reported on the paper, but plus annealing to MSE
bash $SCENES/veach_ajar_noanneal.sh   out_root=$OUT rendering/pos_emb=dense_grid rendering.pos_emb.resolution=128   #The loss reported on the paper, leads to  baised (darker) results
