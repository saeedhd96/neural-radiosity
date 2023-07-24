# In order to render multiple views you need to
# have your new cameras added to the path/to/scene/cameras.xml and set n_views to a number
# or pass a json file containing the transforms here like cameras=transforms.json
# Altenatively, if you wish to render a set of specific views you can specify here, e.g. views=[2,5,6]

python test.py \
    test_rendering.image.spp=512 \
    test_rendering.image.width=512 \
    blocksize=64 \
    experiment=output/nerad_sparse_grid/2023-06-18-04-12-39-cbox \
    n_views=8 \
