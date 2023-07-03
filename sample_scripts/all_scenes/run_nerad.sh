DATA=data
OUT=output/nerad

python watchdog.py --max_retries 120 --  \
    out_root=$OUT \
    'saving=[latest]' \
    batch_size=32768 \
    learning_rate=0.0005 \
    rendering.spp=64 \
    validation.image.step_size=1000 \
    validation.image.first_step=true \
    saving.latest.step_size=1000 \
    n_steps=30000 \
    lr_decay_start=10000 \
    lr_decay_rate=0.35 \
    lr_decay_steps=10000 \
    lr_decay_min_rate=0.01 \
    ${@}
