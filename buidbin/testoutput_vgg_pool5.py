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
m=25088
n=4624
net = caffe.Net(caffe_root + 'models/vgg16/deploy.prototxt',caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel',caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  
transformer.set_transpose('data', (2,0,1))  
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel  
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]  
transformer.set_channel_swap('data', (2,1,0)) # the reference model has channels in BGR order instead of RGB  
 # set net to batch size of 1 即输入为一张图片  
filename='/home/sjq/image/1.jpg'
inImage=Image.open(filename)
out=inImage.resize((224,224),Image.ANTIALIAS)
out.save('/home/sjq/tempImage.jpg')
net.blobs['data'].data[...]  = transformer.preprocess('data', caffe.io.load_image('/home/sjq/tempImage.jpg'))
out=net.forward()
pool5Data=net.blobs['pool5'].data
pool5Data.shape =-1
print type(pool5Data[0])
print pool5Data.shape[0]



