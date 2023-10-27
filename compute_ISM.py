import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math
import shutil
import matplotlib.pyplot as plt
import wandb
import numpy as np
import time
from datetime import datetime
import random

import seaborn as sns
%matplotlib inline
import logging
from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
#os.environ['TPU_LOAD_LIBRARY']='0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf

import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
from scipy.stats.stats import pearsonr  
from scipy.stats.stats import spearmanr  

import enformer_vanilla as enformer

from scipy import stats

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=sys.argv[1])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy=\
        tf.data.experimental.AutoShardPolicy.DATA
    options.deterministic=False
    options.experimental_threading.max_intra_op_parallelism=1
    mixed_precision.set_global_policy('mixed_bfloat16')
    #options.num_devices = 64

    BATCH_SIZE_PER_REPLICA = 1 # batch size 24, use LR ~ 2.5 e -04
    NUM_REPLICAS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS

    
    def deserialize(serialized_example,input_length=196608):
        """Deserialize bytes stored in TFRecordFile."""
        feature_map = {
          'sequence': tf.io.FixedLenFeature([], tf.string),
          'base_pos': tf.io.FixedLenFeature([], tf.string),
          'base_id': tf.io.FixedLenFeature([], tf.string),
          'wt_seq': tf.io.FixedLenFeature([], tf.string),
        }

        data = tf.io.parse_example(serialized_example, feature_map)

        shift = 5
        input_seq_length = input_length + max_shift
        interval_end = input_length + shift

        ### rev_comp
        #rev_comp = random.randrange(0,2)

        example = tf.io.parse_example(serialized_example, feature_map)
        sequence = tf.io.decode_raw(example['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (input_length, 4))
        sequence = tf.cast(sequence, tf.float32)
        
        
        wt = tf.io.decode_raw(example['wt_seq'], tf.bool)
        wt = tf.reshape(wt, (input_length, 4))
        wt = tf.cast(wt, tf.float32)
        
        base_pos = tf.io.parse_tensor(example['base_pos'],
                                      out_type=tf.int32)
        base_id = tf.io.parse_tensor(example['base_id'],
                                      out_type=tf.int32)
    

    files = tf.data.Dataset.list_files(sys.argv[2],shuffle=True, seed=42)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize(record,
                                                     196608),
                                  deterministic=False,
                                  num_parallel_calls=num_parallel)

    dataset=dataset.repeat(1).batch(4).prefetch(1)
    dataset_dist= strategy.experimental_distribute_dataset(dataset)
    dataset_dist_it = iter(dataset_dist)
    
    print(next(dataset_dist_it))