# Uses multi-resolution hash-grid of Muller et al 2022
OUT=output/nerad_hash_grid
SCENES=sample_scripts/all_scenes

bash $SCENES/cbox.sh                  out_root=$OUT
bash $SCENES/copperman.sh             out_root=$OUT
bash $SCENES/bedroom.sh               out_root=$OUT
bash $SCENES/veach_ajar.sh            out_root=$OUT     #artifacts
bash $SCENES/veach_ajar_mit.sh        out_root=$OUT
bash $SCENES/dining_room.sh           out_root=$OUT
bash $SCENES/living_room.sh           out_root=$OUT
bash $SCENES/chair.sh                 out_root=$OUT
bash $SCENES/chair_mirror.sh          out_root=$OUT
bash $SCENES/rings.sh                 out_root=$OUT
bash $SCENES/veach_ajar_anneal.sh     out_root=$OUT  #The loss reported on the paper, but plus annealing to MSE
bash $SCENES/veach_ajar_noanneal.sh   out_root=$OUT  #The loss reported on the paper, leads to  baised (darker) results
