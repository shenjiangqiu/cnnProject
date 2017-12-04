#!/usr/bin/python
#coding=utf-8
# -*- coding: utf-8 -*-
import numpy as np
import caffe 
import sys
import struct
import os
from PIL import Image
caffe_root='/home/sjq/caffe/'
choseImages=os.listdir('/home/sjq/image')
caffe.set_mode_gpu()
m=9216
n=11772
picsize=227
net = caffe.Net(caffe_root + 'models/bvlc_alexnet/deploy.prototxt',caffe_root + 'models/bvlc_alexnet/bvlc_alexnet_old.caffemodel',caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  
transformer.set_transpose('data', (2,0,1))  
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel  
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]  
transformer.set_channel_swap('data', (2,1,0)) # the reference model has channels in BGR order instead of RGB  
 # set net to batch size of 1 即输入为一张图片   
result=[0]*n
for i in xrange(0,n):
    result[i]=[0]*m

for i in xrange(0,n):
    filename='/home/sjq/image/'+str(i+1)+'.jpg'
    net.blobs['data'].data[...]  = transformer.preprocess('data', caffe.io.load_image(filename))
    out=net.forward()
    pool5Data=net.blobs['pool5'].data
    pool5Data.shape = -1
    for j in xrange(0,m):
        result[i][j]=pool5Data[j]

    print i




np.array(result).tofile("/home/sjq/alex_old_pool5.bin")
exit(0)


