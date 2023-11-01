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
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import time
from datetime import datetime
import random

import logging
from silence_tensorflow import silence_tensorflow

os.environ['TPU_LOAD_LIBRARY']='0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf

import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

import enformer_vanilla as enformer

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=sys.argv[1])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

dnase_indices = [120,177,248]
cage_indices=[4831,5082,5210]

dnase_dict = {120: 'jurkat',177: 'cd8', 248: 'cd4'}
cage_dict = {4831: 'jurkat', 5082: 'cd8', 5210: 'cd4'}


if sys.argv[2] == 'IL2RA':
    path="gs://picard-testing-176520/IL2RA_IL15RA/ISM_sequences/IL2RA.tfr"
    start=5964061
    steps = 4548 
elif sys.argv[2] == 'IL15RA':
    path="gs://picard-testing-176520/IL2RA_IL15RA/ISM_sequences/IL15RA.tfr"
    start=5879133
    steps = 3294
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


    model = enformer.Enformer()
    checkpoint_options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint = tf.train.Checkpoint(module=model)#,options=options)
    tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    latest = tf.train.latest_checkpoint("/home/javed/ism_compute/sonnet_weights")
    checkpoint.restore(latest,options=checkpoint_options).assert_existing_objects_matched()
    
    def deserialize(serialized_example,input_length=196607):
        """Deserialize bytes stored in TFRecordFile."""
        feature_map = {
          'sequence': tf.io.FixedLenFeature([],tf.string),
          'base_pos': tf.io.FixedLenFeature([], tf.string),
          'base_id': tf.io.FixedLenFeature([], tf.string),
          'wt_seq': tf.io.FixedLenFeature([], tf.string),
        }

        data = tf.io.parse_example(serialized_example, feature_map)


        ### rev_comp
        #rev_comp = random.randrange(0,2)

        example = tf.io.parse_example(serialized_example, feature_map)
        
        sequence=tf.io.decode_raw(example['sequence'],tf.bool)
        sequence=tf.reshape(sequence,(input_length,4))
        sequence=tf.cast(sequence,tf.float32)
       
        wt = tf.io.decode_raw(example['wt_seq'],tf.bool)
        wt = tf.reshape(wt,(input_length,4))
        wt = tf.cast(wt,tf.float32)

        base_pos = tf.io.parse_tensor(example['base_pos'],
                                      out_type=tf.int32)
        base_id = tf.io.parse_tensor(example['base_id'],
                                      out_type=tf.int32)
 

        rev_sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        rev_sequence = tf.reverse(rev_sequence, axis=[0])

        rev_wt = tf.gather(wt, [3, 2, 1, 0], axis=-1)
        rev_wt = tf.reverse(rev_wt, axis=[0])



        return {
                'sequence':tf.cast(tf.ensure_shape(sequence,[input_length,4]),dtype=tf.float32),
                'rev_sequence': tf.cast(tf.ensure_shape(rev_sequence,[input_length,4]),dtype=tf.float32),
                'wt': tf.cast(tf.ensure_shape(wt,[input_length,4]),dtype=tf.float32),
                'rev_wt':tf.cast(tf.ensure_shape(rev_wt,[input_length,4]),dtype=tf.float32),
                'base_pos': tf.cast(base_pos,dtype=tf.int32),
                'base_id': tf.cast(base_id,dtype=tf.int32)}

    files = tf.data.Dataset.list_files(path,shuffle=True, seed=42)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=4)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize(record,
                                                     196608),
                                  deterministic=False,
                                  num_parallel_calls=4)

    dataset=dataset.repeat(2).batch(8).prefetch(1)
    dataset_dist= strategy.experimental_distribute_dataset(dataset)
    dataset_dist_it = iter(dataset_dist)



    @tf.function
    def val_step(inputs):
        output_seq = model(inputs['sequence'], is_training=False)['human']
        output_rev_seq = tf.reverse(model(inputs['rev_sequence'], is_training=False)['human'],axis=[1])
        output_wt = model(inputs['wt'], is_training=False)['human']
        output_rev_wt = tf.reverse(model(inputs['rev_wt'], is_training=False)['human'],axis=[1])
        
        output_seq_ave = (output_seq+output_rev_seq)/2.0
        output_wt_ave = (output_wt+output_rev_wt)/2.0
   
        base_pos = inputs['base_pos']
        dnase_pos = ((base_pos) - start)//128
        # Create the offsets tensor
        offsets = tf.constant([-3,-2, -1, 0, 1, 2,3], dtype=tf.int32)
        dnase_sub_inds = dnase_pos + offsets

        dnase_seq = tf.gather(output_seq_ave,dnase_indices,axis=2)
        dnase_seq = tf.gather(dnase_seq,dnase_sub_inds,axis=1)
        dnase_seq = tf.reduce_sum(dnase_seq,axis=1)
        dnase_wt = tf.gather(output_wt_ave,dnase_indices,axis=2)
        dnase_wt = tf.gather(dnase_wt,dnase_sub_inds,axis=1)
        dnase_wt = tf.reduce_sum(dnase_wt,axis=1)

        cage_sub_indices = tf.constant([446,447,448,449,450],dtype=tf.int32)
        cage_seq = tf.gather(output_seq_ave,cage_indices,axis=2)
        cage_seq = tf.gather(cage_seq,cage_sub_indices,axis=1)
        cage_seq = tf.reduce_sum(cage_seq,axis=1)

        cage_wt = tf.gather(output_wt_ave,cage_indices,axis=2)
        cage_wt = tf.gather(cage_wt,cage_sub_indices,axis=1)
        cage_wt = tf.reduce_sum(cage_wt,axis=1)

        base_id = inputs['base_id']

        repeat = tf.constant([0,0,0])
        base_id = base_id + repeat
        base_pos = base_pos + repeat
        
        dnase_tensor = tf.constant(dnase_indices,dtype=tf.int32)
        cage_tensor = tf.constant(cage_indices,dtype=tf.int32)

        return tf.squeeze(dnase_seq),tf.squeeze(dnase_wt),tf.squeeze(cage_seq),tf.squeeze(cage_wt),base_id,base_pos,dnase_tensor,cage_tensor

    dnase_mut_list = []
    dnase_wt_list = []
    cage_mut_list = []
    cage_wt_list = []
    base_pos_list = []
    base_id_list = []
    dnase_data_list = []
    cage_data_list = []

    for _ in tf.range(steps): ## for loop within @tf.fuction for improved TPU performance
        start_time = time.time()
        dnase_mut,dnase_wt,cage_mut,cage_wt,base,base_pos,dnase_tensor,cage_tensor=strategy.run(val_step,
                     args=(next(dataset_dist_it),))
        for x in strategy.experimental_local_results(dnase_mut):
            dnase_mut_list.append(tf.reshape(x, [-1]))

        for x in strategy.experimental_local_results(dnase_wt):
            dnase_wt_list.append(tf.reshape(x, [-1]))

        for x in strategy.experimental_local_results(cage_mut):
            cage_mut_list.append(tf.reshape(x, [-1]))

        for x in strategy.experimental_local_results(cage_wt):
            cage_wt_list.append(tf.reshape(x, [-1]))

        for x in strategy.experimental_local_results(base):
            base_id_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(base_pos):
            base_pos_list.append(tf.reshape(x, [-1]))

        for x in strategy.experimental_local_results(dnase_tensor):
            dnase_data_list.append(tf.reshape(x, [-1]))

        for x in strategy.experimental_local_results(cage_tensor):
            cage_data_list.append(tf.reshape(x, [-1]))

    dnase_mut_np=tf.concat(dnase_mut_list,axis=0).numpy()
    dnase_wt_np=tf.concat(dnase_wt_list,axis=0).numpy()
    cage_mut_np=tf.concat(cage_mut_list,axis=0).numpy()
    cage_wt_np=tf.concat(cage_wt_list,axis=0).numpy()
    base_pos_np=tf.concat(base_pos_list,axis=0).numpy()
    base_id_np=tf.concat(base_id_list,axis=0).numpy()

    dnase_np=tf.concat(dnase_data_list,axis=0).numpy()
    cage_np=tf.concat(cage_data_list,axis=0).numpy()

    df = pd.DataFrame(list(zip(base_pos_np, base_id_np, dnase_np,cage_np,dnase_mut_np,dnase_wt_np,cage_mut_np,cage_wt_np)), 
                        columns=['base_pos', 'base_id', 'dnase_idx','cage_idx','dnase_mut','dnase_wt','cage_mut','cage_wt'])
    df['base_pos'] = df['base_pos'] - 1
    df['start'] = df['base_pos']
    df['stop'] =df['start'] + 1
    df['chrom'] = 'chr10'

    df['cage_percent_diff'] = 100.0*(df['cage_mut'] - df['cage_wt']) / df['cage_wt']
    df['dnase_percent_diff'] = 100.0*(df['dnase_mut'] - df['dnase_wt']) / df['dnase_wt']


    # Filter and rename for dnase_idx
    for val in dnase_indices:
        temp_df = df[df['dnase_idx'] == val][['chrom', 'start', 'stop', 'dnase_percent_diff']]
        name = dnase_dict[val]
        full_name = sys.argv[2] + '-DNASE-' + name + ".bedGraph"
        temp_df.to_csv(full_name, index=False,sep='\t',header=False)


    for val in cage_indices:
        temp_df = df[df['cage_idx'] == val][['chrom', 'start', 'stop', 'cage_percent_diff']]
        name = cage_dict[val]
        full_name = sys.argv[2] + '-CAGE-' + name + ".bedGraph"
        temp_df.to_csv(full_name, index=False,sep='\t',header=False)

