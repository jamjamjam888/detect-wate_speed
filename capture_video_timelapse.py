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

#versino確認
version =cv2.__version__
print(version)
#2.4.9.1
#captureクラスの呼び出し
cap = cv2.VideoCapture(0)

########################
t_pre = time.time()
#backgroundを任意のタイミングで撮影する
while True:
    ret, frame = cap.read()
    """
    #配列を確認
    img_array = np.asarray(frame) #numpyで扱える配列をつくる
    print(img_array)
    print(img_array.shape)
    """
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("background_capture", gray_background)

    k = cv2.waitKey(1)&0xff # キー入力を待つ
    if k == ord('p'):
        # 「p」キーで画像を保存
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = "/home/pi/" + "background" + date + ".png"
        cv2.imwrite(path, frame) # ファイル保存

        cv2.imshow(path, frame) # キャプチャした画像を表示
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
print("背景撮影完了")
###########################################################################################

#backgroundを読み込む
background = cv2.imread("/home/pi/background" +date+ ".ping",1)
#gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

#移動量を書き込むテキストファイルを生成し日付を書き込む
f = open("/home/pi/vector_info_"+date+".txt","w")
f.write(str(date)+'\n')
f.close()

#フレーム間差分を計算
cap = cv2.VideoCapture(0)

###動画撮影設定###
"""
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps:",fps)
"""
Width = int(cap.get(3))
Height = int(cap.get(4))

#print("(Width,Height):",Width,Heihgt)

#コーデックを定義しVideoWriter Objectを生成
fourcc = cv2.cv.CV_FOURCC(*"XVID")
out = cv2.VideoWriter("output_"+date+".mp4", fourcc, 50, (Width,Height))

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

    

    #################################################################
    #重心を計算
    contours, hierarchy = cv2.findContours(black_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("contours",contours)
    #print("hierarchy",hierarchy)
    """
    参照:https://www.pynote.info/entry/opencv-contour-manipulation
    FindContours() で抽出した輪郭に対して行う以下の処理を紹介する。
    輪郭の周囲の長さを計算する。 (cv2.arcLength())
    輪郭の面積を計算する。 (cv2.contourArea())
    輪郭を近似する。 (cv2.approxPolyDP())
    輪郭に外接する長方形を取得する。 (cv2.boundingRect())
    輪郭に外接する回転した長方形を取得する。 (cv2.minAreaRect())
    輪郭に外接する円を取得する。 (cv2.minEnclosingCircle())
    輪郭に外接する三角形を取得する。 (cv2.minEnclosingTriangle())
    輪郭に外接する凸包を取得する。 (cv2.convexHull()/isContourConvex())
    """
    areas = []
    for cnt in contours:#cnt:輪郭#輪郭の数だけループする
        area = cv2.contourArea(cnt)#cv2.contourArea(cnt):領域が占める面積を計算
        if area > 200:#輪郭の面積がthreshold以上の場合、リストに追加する
            epsilon = 0.1*cv2.arcLength(cnt,True)
            #領域を囲む周囲長を計算する
            #第二引数は対象とする領域が閉じている(True)か単なる曲線かを表すフラグ
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            #approx:輪郭の近似を行う
            #第二引数は実際の輪郭と近似輪郭の最大距離を表し近似の精度を表すパラメータ
            areas.append(approx)
    #print(areas)
    #輪郭を近似する
    
    approx_contours = []
    for i, cnt in enumerate(areas):
        # 輪郭の周囲の長さを計算する。
        arclen = cv2.arcLength(cnt, True)
        # 輪郭を近似する。
        approx_cnt = cv2.approxPolyDP(cnt, epsilon=0.005 * arclen, closed=True)
        """
        引数
        curve: (NumPoints, 1, 2) の numpy 配列。輪郭
        epsilon: アルゴリズムで使用する許容距離
        closed: 輪郭が閉じているかどうか
        approxCurve: 引数で結果を受け取る場合、指定する。
        返り値
        approxCurve: 近似した輪郭
        """
        approx_contours.append(approx_cnt)
        
        #重心を描画
    ball_pos = []
    for i in range(len(approx_contours)):  #重心位置を計算
        count = len(approx_contours[i])
        area = cv2.contourArea(approx_contours[i])  #面積計算
        x, y = 0.0, 0.0
        for j in range(count):
            x += approx_contours[i][j][0][0]
            y += approx_contours[i][j][0][1]

        x /= count
        y /= count
        x = int(x)
        y = int(y)
        ball_pos.append([x, y])
        #重心座標を書き込む
        ball_position = (x,y)
        cv2.putText(frame, str(0), ball_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))        
        
        
    np_ball_pos = np.array(ball_pos)
    print("moment:\n"+str(np_ball_pos))
    
    
    

#######################calcurate diff#######################################
    #ball_pre
    print("\n"+"ball_pre"+"\n"+str(ball_pre))
    
    if len(np_ball_pos) == len(ball_pre):
        np_ball_pre = np.array(ball_pre)
        diff = np_ball_pos - np_ball_pre
        print("\n"+"diff"+"\n"+str(diff))
        
        #calcurate time_lapse
        t_now = time.time()
        t_lapse = t_now - t_pre
        
        #calcurate vector
        vector = diff/t_lapse
        print("\n"+"vector"+"\n"+str(vector))
        #write moment + vector on the livevideo
        for number in range(len(np_ball_pos)):
            moment = np_ball_pos[number]
            #cv2.arrowedLine(frame, tuple(np_ball_pre[number]),tuple(np_ball_pos[number]), (0, 0, 255), thickness=1)
            #cv2.drawMarker(frame, tuple(np_ball_pos[number]), (0, 0, 255))
            #probably can't use arrowedLine, drawMarker
            cv2.circle(frame, tuple(np_ball_pos[number]), 15, (0, 0, 255), thickness=1)    
    
    else:
        vector = []
        print("error")
        #only write moment
        for number in range(len(np_ball_pos)):
            moment = np_ball_pos[number]
            #cv2.drawMarker(frame, tuple(np_ball_pos[number]), (0, 0, 255))
            #cv2.circle(frame, tuple(np_ball_pos[number]), 15, (0, 0, 255), thickness=1)
    
    #write vector_info on the livevideo
    position1 = (50,50)
    cv2.putText(frame, str(vector), position1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
        
        
    #write boll_position
    position2 = (50,100)
    cv2.putText(frame, str(ball_pos), position2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

    #get pre_info
    ball_pre = np_ball_pos
    t_pre = time.time()

####################################################
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

        
        
        break
print("output:{}".format(output))
# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
out.release()
#cv2.destroyAllWindows()
print("終了")

################################################

