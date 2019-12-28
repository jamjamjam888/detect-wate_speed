#!/usr/bin/env python
#coding:utf-8

#ラズパイカメラをマウントしてusbカメラにするコマンド
#sudo modprobe bcm2835-v4l2

#usbカメラで動かすことに成功

from time import sleep
import math
import cv2
import numpy as np
from datetime import datetime

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
    #elif k == ord('q'):
        # 「q」キーが押されたら終了する
        break
# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
print("背景撮影完了")
###########################################################################################

#backgroundを読み込む
background = cv2.imread("/home/pi/background" +date+ ".ping",1)
#gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

#フレーム間差分を計算
cap = cv2.VideoCapture(0)
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
    retval, black_diff = cv2.threshold(color_diff_ini, 130, 255, cv2.THRESH_BINARY)
    
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
    #輪郭を近似する
    approx_contours = []
    for i, cnt in enumerate(contours):
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
    print(approx_contours)
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
    print("ball_pos",ball_pos)
    #######################
    for pos in range(len(ball_pos))
    #加工なし画像を表示する
    cv2.imshow('Moment Frame', frame)
    #マスクをかける

    #bitwise_and = cv2.bitwise_and(gray1, black_diff)
    #加工ありの画像を表示    
    #cv2.imshow('bitwise_and',bitwise_and)

    #キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
print("終了")
################################################
