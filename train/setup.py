# from kaggle_datasets import KaggleDatasets
# GCS_PATH = KaggleDatasets().get_gcs_path('ai4code')
GCS_PATH = 'gs://kds-efc392fe4d47b8bcfc0dc0bf9f8789398c4764931d581ec8c7941fca'

import tensorflow as tf
try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
    BATCH_SIZE = 64 * STRATEGY.num_replicas_in_sync
except Exception:
    TPU = None
    STRATEGY = tf.distribute.get_strategy()
    BATCH_SIZE = 8

if TPU is not None:
    print("Using TPU v3-8")
else:
    print("Using GPU/CPU")