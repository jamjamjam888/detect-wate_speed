
import time
from time import sleep
import math
import cv2
import numpy as np
from datetime import datetime

cap = cv2.VideoCapture(0)

class Raspicam:
    def background_capture(self):       
        #backgroundを任意のタイミングで撮影する
        while True:
            ret, frame = cap.read()
            gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("background_capture", gray_background)

            k = cv2.waitKey(1)&0xff # キー入力を待つ
            if k == ord('p'):
            # 「p」キーで画像を保存
                date = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(date)
                path = "/home/pi/" + "background" + str(date)+ ".png"
                cv2.imwrite(path, frame) # ファイル保存

                #cv2.imshow(path, frame) # キャプチャした画像を表示
                break
                

        # キャプチャをリリースして、ウィンドウをすべて閉じる
        cap.release()
        cv2.destroyAllWindows()
        print("背景撮影完了")
        return str(date)


    def calcurate(self, date, fps):
        
        #backgroundを読み込む
        background = cv2.imread("/home/pi/background" +date+ ".png",1)
        gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        
        #カメラ設定
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, fps)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("fps:",fps)
        
        count = 1
        
        while (True):
            
            
            ret, frame = cap.read()
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #加工ありの画像を表示
            #cv2.imshow('Gray Frame',gray1)
            #差分検出
            color_diff_ini = cv2.absdiff(gray1, gray_background)
            #閾値処理
            retval, black_diff = cv2.threshold(color_diff_ini, 80, 255, cv2.THRESH_BINARY)
            #加工ありの画像を表示
            cv2.imshow('black_diff',black_diff)
            #重心を計算
            contours, hierarchy = cv2.findContours(black_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
            if count != 1:
                #ball_pre
                print("\n"+"ball_pre"+"\n"+str(ball_pre))
            
                #速度を格納するための配列を生成する
                vector = []
                #画像上での検知した個数がフレームの前後で同じときだけ移動量を計算する
                if len(np_ball_pos) == len(ball_pre):
                    np_ball_pre = np.array(ball_pre)
                #画像上での移動量。Δx=diff[0],Δy=diff[1]
                    diff = np_ball_pos - np_ball_pre
                    print("\n"+"diff"+"\n"+str(diff))
                #calcurate time_lapse
                #fpsを計算する。fpsを自分で設定してもよい。今回はfps=30で行ったのでそれを用いた
                #t_now = time.time()
                #t_lapse = t_now - t_pre
            
                    if len(diff) != 0:#配列の中身が空じゃないとき
                    #calcurate vector
                    #280mmm:58pxl = 4.8mm:1pxl
                    #現実での移動量=比例定数*画像上での移動量.
                    #ただし、高さ28cm
                        diff_list = diff.tolist()
                        print("diff_list",diff_list)
                        real_x = round(diff_list[0][0]*4.8)
                        real_y = round(diff_list[0][1]*4.8)
                        print(real_x,real_y)
                    #絶対値の2乗計算
                        x_abs = abs(real_x)
                        y_abs = abs(real_y)
                        cal = x_abs**2 + y_abs**2
                    #√abs*fps
                        vector = (round(np.sqrt(cal)))
                        print("\n"+"vector"+"\n"+str(vector))
                    #write moment + vector on the livevideo
                        for number in range(len(np_ball_pos)):
                            moment = np_ball_pos[number]
                            cv2.circle(frame, tuple(np_ball_pos[number]), 15, (0, 0, 255), thickness=1)
            
                else:
                    print("error")
                    #only write moment
                    for number in range(len(np_ball_pos)):
                        moment = np_ball_pos[number]
            
                #write vector_info on the livevideo
                position1 = (50,50)
                cv2.putText(frame, str(vector), position1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            
                #write boll_position
                position2 = (50,100)
                cv2.putText(frame, str(ball_pos), position2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
            
                #get pre_info
                ball_pre = np_ball_pos
            else:
                ball_pre = np_ball_pos
            ####################################################
            #加工なし画像を表示する
            cv2.imshow('Moment Frame', frame)
            
            count += 1
            
            k = cv2.waitKey(1)&0xff # キー入力を待つ
            if k == ord('p'):
                # 「p」キーで画像を保存
                date = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = "/home/pi/" + "moment" + date + ".png"
                cv2.imwrite(path, frame) # ファイル保存
            
                # キャプチャをリリースして、ウィンドウをすべて閉じる
                cap.release()
                #out.release()
                cv2.destroyAllWindows()
                break
