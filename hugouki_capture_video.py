#参照:https://helloidea.org/index.php/archives/1925.html

#!/usr/bin/env python
#coding:utf-8

#(備考)
#cv2.getBuildInformation()でViedoI/Oが有効化されているか確認。「FFMPEG」が有効ならok!


import time
from time import sleep
import math
import cv2
import numpy as np
from datetime import datetime

from def_raspicamera import Raspicam

#version確認
version =cv2.__version__
print(version)

raspi = Raspicam()
#rascam.background_capture()

#移動量を書き込むテキストファイルを生成し日付を書き込む
date = raspi.background_capture()

print(date)
f = open("/home/pi/vector_info_"+date+".txt","w")
f.write(str(date)+'\n')
f.close()

#distance_lapse
output = []
ball_pre = []
fps = 1
raspi.calcurate(date, fps)

################################################
