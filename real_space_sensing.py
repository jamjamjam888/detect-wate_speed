#!/usr/bin/env python
#coding:utf-8

#(備考)
#cv2.getBuildInformation()でViedoI/Oが有効化されているか確認。「FFMPEG」が有効ならok!
#usbカメラをマウント
#usbls


import time
from time import sleep
import math
import cv2
import numpy as np
from datetime import datetime

#versino確認
version =cv2.__version__
print(version)
#2.4.9.1
#captureクラスの呼び出し

#Build Info
#print(cv2.getBuildInformation())

#windows7でusbカメラを使用する場合だったら2を指定
cap = cv2.VideoCapture(2)

###動画撮影設定###

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
Width = int(cap.get(3))
Height = int(cap.get(4))
print("(Width,Heihgt):",Width,Height)
windowsize = (800, 800)
########################
t_pre = time.time()
#backgroundを任意のタイミングで撮影する
while True:
    #sleep(1)
    ret, frame = cap.read()
    #print(ret)
    
   
    #frameサイズ変更
    #frame = cv2.resize(frame, dsize=(100, 100))
    frame = cv2.resize(frame, windowsize)
    #表示
    cv2.imshow("frame", frame)

    """
    #配列を確認
    img_array = np.asarray(frame) #numpyで扱える配列をつくる
    print(img_array)
    print(img_array.shape)
    """
    #gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("background_capture", gray_background)

    k = cv2.waitKey(1)&0xff # キー入力を待つ
    if k == ord('p'):
        # 「p」キーで画像を保存
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = "C:\Users\owner\Desktop\watanabe\VLC\input\capture" + date + ".png"
        cv2.imwrite(path, frame) # ファイル保存

        cv2.imshow(path, frame) # キャプチャした画像を表示
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
print("背景撮影完了")
###########################################################################################

#backgroundを読み込む
background = cv2.imread("C:\Users\owner\Desktop\watanabe\VLC\input\capture" +date+ ".ping",1)
#gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

#実空間センシングのrealtime処理を行う
cap = cv2.VideoCapture(2)

###動画撮影設定###
Width = int(cap.get(3))
Height = int(cap.get(4))
print("(Width,Heihgt):",Width,Height)
#print("(Width,Height):",Width,Heihgt)

#fps
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps:",fps)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4','v')
out = cv2.VideoWriter("output_"+date+".mp4", fourcc, fps, (Width,Height))

###参照###
    # CV_FOURCC('D','I','B',' ')    = 無圧縮
    # CV_FOURCC('P','I','M','1')    = MPEG-1 codec
    # CV_FOURCC('M','J','P','G')    = motion-jpeg codec (does not work well)
    # CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
    # CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
    # CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
    # CV_FOURCC('U', '2', '6', '3') = H263 codec
    # CV_FOURCC('I', '2', '6', '3') = H263I codec
    # CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec
########

#distance_lapse
output = []
ball_pre = []
while (True):

    #VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()
    """
    #配列を確認
    print("配列を確認")
    print(type(frame))
    img_array = np.asarray(frame) #numpyで扱える配列をつくる
    print(img_array)
    print(img_array.shape)
    """
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #加工ありの画像を表示    
    #cv2.imshow('Gray Frame',gray1)
    
    #差分検出
    color_diff_ini = cv2.absdiff(gray1, gray_background)
    #閾値処理
    retval, black_diff = cv2.threshold(color_diff_ini, 80, 255, cv2.THRESH_BINARY)
    """
    #write video on raspi
    out.write(black_diff)
    """
    #加工ありの画像を表示    
    cv2.imshow('black_diff',black_diff)

    

    #write video on raspi
    out.write(frame)
    
    #加工なし画像を表示する
    cv2.imshow('Moment Frame', frame)
    
    #save video
    #video.write(frame)

    #bitwise_and = cv2.bitwise_and(gray1, black_diff)
    #加工ありの画像を表示    
    #cv2.imshow('bitwise_and',bitwise_and)
    
    #vectorをtexifielに書き込む
    #textfile作成
    f = open("/home/pi/vector_info_"+date+".txt","a")
    f.write(str(vector)+'\n')
    f.close()

    #キー入力を1ms待って、k がpだったらBreakする
    k = cv2.waitKey(100)&0xff # キー入力を待つ
    #now = datetime.now()
    #print(str(now)+"\n")
    
    if k == ord('p'):
        # 「p」キーで画像を保存
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = "/home/pi/" + "moment" + date + ".png"
        cv2.imwrite(path, frame) # ファイル保存
        
        print("output:{}".format(output))
        # キャプチャをリリースして、ウィンドウをすべて閉じる
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        break
print("output:{}".format(output))
print("終了")

################################################
