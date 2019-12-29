#!/usr/bin/env python
#coding:utf-8

#ラズパイカメラをマウントしてusbカメラにするコマンド
#sudo modprobe bcm2835-v4l2

#usbカメラで動かすことに成功
import time
from time import sleep
import math
import cv2
import numpy as np
from datetime import datetime

version =cv2.__version__
print(version)
#2.4.9.1

cap = cv2.VideoCapture(0)

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
#cv2.destroyAllWindows()
print("背景撮影完了")
###########################################################################################

#backgroundを読み込む
background = cv2.imread("/home/pi/background" +date+ ".ping",1)
#gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

#フレーム間差分を計算
cap = cv2.VideoCapture(0)

#distance_lapse
output = []

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
    retval, black_diff = cv2.threshold(color_diff_ini, 50, 255, cv2.THRESH_BINARY)
    
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
        if area > 300:#輪郭の面積が5000以上の場合、リストに追加する
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
        # 元の輪郭及び近似した輪郭の点の数を表示する。
        #print("contour {}: {} -> {}".format(i, len(cnt), len(approx_cnt)))
    #print(approx_contours)
    #流体を描画
    cv2.drawContours(frame,approx_contours,-1,(0,0,255),1)
    
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
    print("ball_pos"+str(ball_pos)+'\n')

#######################
    print("重心座標")
    for number in range(len(ball_pos)):
        moment = ball_pos[number]
        print(moment)
        
        #calcurate velocity
        if not moment:
            output.append(moment)
            #output[-2] = pre_value
            #output = [.... [-2] [-1]]
            #                pre  now
            if len(output) >= 2:
                x_distance_diff = output[-1][0] - output[-2][0]
                y_distance_diff = output[-1][1] - output[-2][1]
                #time_lapse = fps

        cv2.circle(frame, tuple(ball_pos[number]), 15, (0, 0, 255), thickness=1)
    
####################################################
    #calcaurate ve;osity
        
    #get fps
    #fps = cap.get(cv2.CAP_PROP_FPS)

            #x_valocity = x_distance_diff/time_lapse
            #y_valocity = y_distance_diff/time_lapse
            #velocity = [x_velocity, y_velocity]
            #valocity_data =[]
            #velocity_data.append(velocity)


    
    
    #加工なし画像を表示する
    cv2.imshow('Moment Frame', frame)
    #マスクをかける

    #bitwise_and = cv2.bitwise_and(gray1, black_diff)
    #加工ありの画像を表示    
    #cv2.imshow('bitwise_and',bitwise_and)

    #キー入力を1ms待って、k がpだったらBreakする
    k = cv2.waitKey(1)&0xff # キー入力を待つ
    if k == ord('p'):
        # 「p」キーで画像を保存
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = "/home/pi/" + "moment" + date + ".png"
        cv2.imwrite(path, frame) # ファイル保存

        cv2.imshow(path, frame) # キャプチャした画像を表示
        
        break
print("output:{}".format(output))
# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
#cv2.destroyAllWindows()
print("終了")
################################################
