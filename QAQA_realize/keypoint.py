# 提取单个视频的姿态特征，生成json文件
import json

import pandas as pd
import copy
import cv2
import os
import mediapipe as mp
import time

# print(str(now))
# from google.colab.patches import cv2_imshow
import numpy as np


def get_lableandwav(path, dir):
    allpath = []
    dirs = os.listdir(path)
    for a in dirs:
        # print(a)
        # print(os.path.isfile(path+"/"+a))
        if os.path.isfile(path + "/" + a):
            allpath.append(dirs)
    return allpath


def obtain_keyponint(action_path):
    s_time = time.time()
    # mp.solutions.drawing_utils用于绘制
    mp_drawing = mp.solutions.drawing_utils

    # 参数：1、颜色，2、线条粗细，3、点的半径
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

    # mp.solutions.pose，是人的骨架
    mp_pose = mp.solutions.pose

    # 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
    pose_mode = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    list_str = ['nose', 'leftEyeInner', 'leftEye', 'leftEyeOUter', 'rightEyeInner', 'rightEye', 'rightEyeOuter',
                'leftEar', 'rightEar', 'mouthLeft', 'mouthRight', 'leftShoulder', 'rightShoulder',
                'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftPinky', 'rightPinky',
                'leftIndex', 'rightIndex', 'leftThumb', 'RightThumb', 'leftHip', 'rightHip', 'leftKnee',
                'rightKnee', 'leftAnkle', 'rightAnkle', 'leftHeel', 'rightHeel', 'leftFootIndex', 'rightFootIndex']

    # action_path = 'E:/01 体育/03 diving/clip_video/test/gym1_11.mp4'

    name = action_path[0:-3] + 'csv'

    save_path = './data'

    path_cap = action_path
    cap = cv2.VideoCapture(path_cap)
    # print(path_cap)
    cnt = -1
    df_res = pd.DataFrame(None, columns=list_str)
    # df_res.loc[0] = 0

    # 获取视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total_frames = {0}".format(total_frames))
    # 计算每秒抽取的帧数
    frames_per_second = 5
    frame_interval = int(round(fps) / frames_per_second)
    if frame_interval < 1:
        frame_interval = 30
    if total_frames > 2000:
        frame_interval = 15
    print("frame_interval = {0}".format(frame_interval))
    # 创建一个计数器来跟踪当前帧数
    frame_count = 0

    while cap.isOpened():
        frame_count += 1
        success, image = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            break
        if frame_count % frame_interval != 0:
            continue
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像
        results = pose_mode.process(image1)

        '''
      mp_holistic.PoseLandmark类中共33个人体骨骼点
      '''

        # 绘制
        # mp_drawing.draw_landmarks(
        #    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

        # cv2_imshow(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cnt = cnt + 1
        # df_res.loc[cnt] = 0
        if results.pose_landmarks:
            for index, landmarks in enumerate(results.pose_landmarks.landmark):
                # print(list_str[index])
                cx = landmarks.x
                cy = landmarks.y
                cz = landmarks.z
                v = landmarks.visibility

                tmp_list = list([cx, cy, cz, v])
                # print(tmp_list)
                # df_res[list_str[index]].loc[cnt] = (tmp_list)
                df_res.at[cnt, list_str[index]] = tmp_list
                # print(list_str[index],cnt,tmp_list)
                # time.sleep(2)
                # print(df_res)
                # filename = 'mediapipe_results4.txt'
                # test = open(filename, 'a')

                # if (index <= 0):
                #    print('\n', file=test)
                # print(cx, ',', cy, ',', cz, ',', v, end=',', file=test)

    now = time.time()
    # df_res.to_csv(save_path+'/'+str(now)+'.json', index=None)
    absolute_path = save_path + '/' + str(now) + '.json'
    # df_res.to_json(absolute_path, orient="columns", force_ascii=False)
    pose_mode.close()
    cv2.destroyAllWindows()
    cap.release()

    print("point point point point time = {0}".format(now - s_time))
    return df_res, total_frames


def obtain_keyponint(action_path):
    s_time = time.time()
    # mp.solutions.drawing_utils用于绘制
    mp_drawing = mp.solutions.drawing_utils

    # 参数：1、颜色，2、线条粗细，3、点的半径
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

    # mp.solutions.pose，是人的骨架
    mp_pose = mp.solutions.pose

    # 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
    pose_mode = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    list_str = ['nose', 'leftEyeInner', 'leftEye', 'leftEyeOUter', 'rightEyeInner', 'rightEye', 'rightEyeOuter',
                'leftEar', 'rightEar', 'mouthLeft', 'mouthRight', 'leftShoulder', 'rightShoulder',
                'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftPinky', 'rightPinky',
                'leftIndex', 'rightIndex', 'leftThumb', 'RightThumb', 'leftHip', 'rightHip', 'leftKnee',
                'rightKnee', 'leftAnkle', 'rightAnkle', 'leftHeel', 'rightHeel', 'leftFootIndex', 'rightFootIndex']

    # action_path = 'E:/01 体育/03 diving/clip_video/test/gym1_11.mp4'

    name = action_path[0:-3] + 'csv'

    save_path = './data'

    path_cap = action_path
    cap = cv2.VideoCapture(path_cap)
    # print(path_cap)
    cnt = -1
    df_res = pd.DataFrame(None, columns=list_str)
    # df_res.loc[0] = 0

    # 获取视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total_frames = {0}".format(total_frames))
    # 计算每秒抽取的帧数
    frames_per_second = 5
    frame_interval = int(round(fps) / frames_per_second)
    if frame_interval < 1:
        frame_interval = 30
    if total_frames > 2000:
        frame_interval = 15
    print("frame_interval = {0}".format(frame_interval))
    # 创建一个计数器来跟踪当前帧数
    frame_count = 0

    while cap.isOpened():
        frame_count += 1
        success, image = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            break
        if frame_count % frame_interval != 0:
            continue
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像
        results = pose_mode.process(image1)

        '''
      mp_holistic.PoseLandmark类中共33个人体骨骼点
      '''

        # 绘制
        # mp_drawing.draw_landmarks(
        #    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

        # cv2_imshow(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cnt = cnt + 1
        # df_res.loc[cnt] = 0
        if results.pose_landmarks:
            for index, landmarks in enumerate(results.pose_landmarks.landmark):
                # print(list_str[index])
                cx = landmarks.x
                cy = landmarks.y
                cz = landmarks.z
                v = landmarks.visibility

                tmp_list = list([cx, cy, cz, v])
                # print(tmp_list)
                # df_res[list_str[index]].loc[cnt] = (tmp_list)
                df_res.at[cnt, list_str[index]] = tmp_list
                # print(list_str[index],cnt,tmp_list)
                # time.sleep(2)
                # print(df_res)
                # filename = 'mediapipe_results4.txt'
                # test = open(filename, 'a')

                # if (index <= 0):
                #    print('\n', file=test)
                # print(cx, ',', cy, ',', cz, ',', v, end=',', file=test)

    now = time.time()
    # df_res.to_csv(save_path+'/'+str(now)+'.json', index=None)
    absolute_path = save_path + '/' + str(now) + '.json'
    # df_res.to_json(absolute_path, orient="columns", force_ascii=False)
    pose_mode.close()
    cv2.destroyAllWindows()
    cap.release()

    print("point point point point time = {0}".format(now - s_time))
    return df_res, total_frames


def obtain_keyponint_api(action_video):
    # mp.solutions.drawing_utils用于绘制
    mp_drawing = mp.solutions.drawing_utils

    # 参数：1、颜色，2、线条粗细，3、点的半径
    DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
    DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

    # mp.solutions.pose，是人的骨架
    mp_pose = mp.solutions.pose

    # 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
    pose_mode = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    list_str = ['nose', 'leftEyeInner', 'leftEye', 'leftEyeOUter', 'rightEyeInner', 'rightEye', 'rightEyeOuter',
                'leftEar', 'rightEar', 'mouthLeft', 'mouthRight', 'leftShoulder', 'rightShoulder',
                'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftPinky', 'rightPinky',
                'leftIndex', 'rightIndex', 'leftThumb', 'RightThumb', 'leftHip', 'rightHip', 'leftKnee',
                'rightKnee', 'leftAnkle', 'rightAnkle', 'leftHeel', 'rightHeel', 'leftFootIndex', 'rightFootIndex']

    # action_path = 'E:/01 体育/03 diving/clip_video/test/gym1_11.mp4'

    # name = action_path[0:-3] + 'csv'

    save_path = './data'

    # path_cap = action_path
    cap = cv2.VideoCapture(action_video)
    # print(path_cap)
    cnt = -1
    df_res = pd.DataFrame(None, columns=list_str)
    # df_res.loc[0] = 0

    # 获取视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps = {0}".format(fps))
    print("total_frames = {0}".format(total_frames))
    # 计算每秒抽取的帧数
    frames_per_second = 10
    frame_interval = int(fps / frames_per_second)
    if frame_interval < 1:
        frame_interval = 1
    print("frame_interval = {0}".format(frame_interval))
    # 创建一个计数器来跟踪当前帧数
    frame_count = 0

    while cap.isOpened():
        frame_count += 1
        success, image = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            break
        if frame_count % frame_interval != 0:
            continue
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理RGB图像
        results = pose_mode.process(image1)

        '''
      mp_holistic.PoseLandmark类中共33个人体骨骼点
      '''

        # 绘制
        # mp_drawing.draw_landmarks(
        #    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

        # cv2_imshow(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cnt = cnt + 1
        # df_res.loc[cnt] = 0
        if results.pose_landmarks:
            for index, landmarks in enumerate(results.pose_landmarks.landmark):
                # print(list_str[index])
                cx = landmarks.x
                cy = landmarks.y
                cz = landmarks.z
                v = landmarks.visibility

                tmp_list = list([cx, cy, cz, v])
                # print(tmp_list)
                # df_res[list_str[index]].loc[cnt] = (tmp_list)
                df_res.at[cnt, list_str[index]] = tmp_list
                # print(list_str[index],cnt,tmp_list)
                # time.sleep(2)
                # print(df_res)
                # filename = 'mediapipe_results4.txt'
                # test = open(filename, 'a')

                # if (index <= 0):
                #    print('\n', file=test)
                # print(cx, ',', cy, ',', cz, ',', v, end=',', file=test)
    now = time.time()

    # df_res.to_csv(save_path+'/'+str(now)+'.json', index=None)
    # absolute_path = save_path + '/' + str(now) + '.json'
    # df_res.to_json(absolute_path, orient="columns", force_ascii=False)
    pose_mode.close()
    cv2.destroyAllWindows()
    cap.release()
    return df_res, total_frames, frame_interval


def json_to_dataframe(json_file_path):  # json文件读取3d骨骼关键点坐标转换为dataframe的格式
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 提取列名
    keypoint_id2name = data['meta_info']['keypoint_id2name']
    list_str = [name for _, name in sorted(keypoint_id2name.items(), key=lambda x: int(x[0]))]

    # 提取每一帧的keypoint坐标
    instance_info = data['instance_info']
    frames = {}
    for frame in instance_info:
        frame_id = frame['frame_id']
        keypoints_data = {name: [0, 0, 0] for name in list_str}  # 初始化坐标数据

        for instance in frame['instances']:
            keypoints = instance['keypoints']
            for i, kp in enumerate(keypoints):
                if i < len(list_str):
                    keypoints_data[list_str[i]] = kp

        frames[frame_id] = keypoints_data

    # 创建DataFrame
    df_res = pd.DataFrame.from_dict(frames, orient='index')

    # 获取总帧数
    total_frames = len(frames)

    return df_res, total_frames


if __name__ == '__main__':
    json_file_path = "E:\\FJNU\project\\PoseDetection\\score_v2\\data\\20231214_20231214065155A025.json"
    df, total_frames = json_to_dataframe(json_file_path)
    df.to_csv('E:\\FJNU\project\\PoseDetection\\score_v2\\data\\testdf.csv', index=False)
    print(df)
    print(f"Total Frames: {total_frames}")
