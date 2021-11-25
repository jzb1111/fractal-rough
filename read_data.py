# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 21:29:13 2021

@author: asus
"""

import soundfile
import numpy as np
import os
import tensorflow as tf

def read_wave_data(filename):
    wave_data,framerate=soundfile.read(filename)
    return wave_data, framerate

def file_name(file_dir):
    root_=[]
    dirs_=[]
    files_=[]
    for root, dirs, files in os.walk(file_dir):  
        #print(root) #当前目录路径  
        #print(dirs) #当前路径下所有子目录  
        #print(files) #当前路径下所有非目录子文件
        root_.append(root)
        dirs_.append(dirs)
        files_.append(files)
    return root_,dirs_,files_

def read_text_file(filename):
    with open(filename,"r") as f:
        data=f.read()
        f.close()
    return data

def read_train_file(num):
    
    return 0

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude