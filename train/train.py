if 0**1:
    from utils import *

EPOCHS = 5
with STRATEGY.scope():
    model = get_model()

    optimizer = transformers.AdamWeightDecay(
        learning_rate=ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=steps_per_epoch * EPOCHS,
            decay_rate=0.96,
            staircase=True,
        ),
        weight_decay_rate=0.04,
        exclude_from_weight_decay=[
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
        ],
    )
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

# metrics = model.fit(
#     train_dataset,
#     steps_per_epoch=steps_per_epoch,
#     epochs=EPOCHS,
#     verbose=VERBOSE,
# ).history

print(per_dataset_score(model, val_filenames))

model.save_weights(f"model.h5")
