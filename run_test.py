import numpy as np
from realize import realize
import pandas as pd
from realize.realize_acdtw import realize_dtw
from realize.realize_fastdtw import fastdtw
import sys
from realize.json_read import *
from deal_data import *
from realize.keypoint import json_to_dataframe, obtain_keyponint
import time
import cv2
import random
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

CUT_FRAMES = 10  # 随机抽取视频帧数
TRUE_PERSON = 50  # 人类视频中出现的百分比


def random_frames_from_video(video_path, num_frames=CUT_FRAMES):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return []

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 生成随机帧索引
    frame_indices = random.sample(range(total_frames), min(num_frames, total_frames))

    # 读取随机帧
    frames = []
    for frame_index in frame_indices:
        # 定位到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Error reading frame at index {frame_index}")

    # 释放VideoCapture对象
    cap.release()

    return frames


def ComputeScore(D, x, y):
    score = D(x, y)[0]
    while x != 0 and y != 0:
        x, y = D(x, y)[1], D(x, y)[2]
        score += D(x, y)[0]
    return score


def action_dtw():
    path_test = sys.argv[1]
    path_temp = sys.argv[2]
    # 这是测试视频生成的json路径：
    datax = json_read_(path_test)
    # 这是模板视频生成的json路径：
    datay = json_read_(path_temp)
    datax = deal_data_(datax, path_test)
    datay = deal_data_(datay, path_temp)

    realize_dtw(datax, datay)


def action_keypoint(path_test, path_temp):  # （测试，模板）df_test是获取关节点信息，dataframe是pands的一种数据结构
    print("开始处理视频", flush=True)
    suffix_test = path_test[-4:]
    suffix_temp = path_temp[-4:]
    if suffix_test != 'json':
        df_test, num_frames_test = obtain_keyponint(path_test)
        df_test = data_read_(df_test)
        # print(df_test, flush=True)
        if df_test.empty:
            print("测试视频有问题，获取内容为空", flush=True)
        df_test = deal_data_(df_test, path_test)
        print("处理测试视频完成", flush=True)
    else:  # json->dataframe 最终返回一个df_test和num_frames_test
        test_tmp, num_frames_test = json_to_dataframe(path_test)
        test_tmp = data_read_(test_tmp)
        if test_tmp.empty:
            print("测试文件有问题，获取内容为空", flush=True)
        else:
            df_test = deal_data_(test_tmp, path_test)
        print("处理测试文件完成", flush=True)
    if suffix_temp != 'json':
        df_temp, num_frames_temp = obtain_keyponint(path_temp)
        df_temp = data_read_(df_temp)
        if df_temp.empty:
            print("模板视频有问题，获取内容为空", flush=True)
        start1 = time.time()
        df_temp = deal_data_(df_temp, path_temp)
        end1 = time.time()
        running_time1 = end1 - start1
        print('df_temp = deal_data_处理数据成df的时间 ,time cost : %.5f sec' % running_time1)
    else:  # json->dataframe 最终返回一个df_temp和num_frames_temp
        temp_tmp, num_frames_temp = json_to_dataframe(path_temp)
        temp_tmp = data_read_(temp_tmp)
        if temp_tmp.empty:
            print("模板文件有问题，获取内容为空", flush=True)
        else:
            df_temp = deal_data_(temp_tmp, path_temp)
        print("处理模板文件完成", flush=True)
    return df_test, df_temp, num_frames_test, num_frames_temp


if __name__ == '__main__':
    all_start = time.time()
    # print(time.time())
    # path_test = "E:/FJNU/project/PoseDetection/score_v1/data/gym2_2.mp4"
    # path_temp = "E:/FJNU/project/PoseDetection/score_v1/data/template_2.mp4"
    path_test = "E:/FJNU/project/PoseDetection/score_v2/data/20231214_20231214065155A025.json"
    path_temp = "E:/FJNU/project/PoseDetection/score_v2/data/3.1第一式双手托天理三焦_20231123001625A023.json"
    # 检测人类
    # Reg = PersonReg()
    # random_frames_test = random_frames_from_video(path_test, num_frames=10)
    # isPerson = False
    # PerResult = []
    # for i in random_frames_test:
    #     # print(i.shape)
    #     PerResult.append(Reg.isperson(i))
    #     # print(PerResult)
    # # 计算人类出现的百分比，并根据阈值判断是否继续执行后续流程
    # true_count = PerResult.count(True) * 10
    # print(true_count)
    # total_count = len(PerResult)
    # PersonOccu = (true_count / total_count) * 10
    # if PersonOccu < TRUE_PERSON:
    #     print("请上传人类或是人类占比大于{0}%的视频".format(TRUE_PERSON))
    # else:
    result_str = ""
    # 关键点的提取
    data_x, data_y, num_frames_test, num_frames_temp = action_keypoint(path_test, path_temp)
    # data_x.to_csv("D:/FJNU/project/PoseDetection/score_v2/data/data_x.csv", index=False)
    # data_y.to_csv("D:/FJNU/project/PoseDetection/score_v2/data/data_y.csv", index=False)
    col_name = data_x.columns.tolist()  # 列名，在路径回溯的时候用到
    print("DTW算法开始计算得分", flush=True)
    print(f'测试文件的帧数：{num_frames_test}')
    print(f'模板文件的帧数：{num_frames_temp}')
    data_x_numpy = data_x.to_numpy()
    data_y_numpy = data_y.to_numpy()
    # print(data_x_numpy)
    # 运行DTW算法计算相似度得分
    start = time.time()
    #score, result_str = realize_dtw(data_x_numpy, data_y_numpy, path_test, path_temp, num_frames_test, col_name)
    D = defaultdict(lambda: (float('inf'),))
    D, path = fastdtw(data_x_numpy, data_y_numpy, num_frames_test, col_name)
    score = ComputeScore(D, num_frames_test, num_frames_temp)
    end = time.time()
    running_time = end - start
    all_running_time = end - all_start
    print('realize_dtw time cost : %.5f sec' % running_time)
    print('all time cost : %.5f sec' % all_running_time)
    result_str += "\nfinal score={0}".format(score)

    print(result_str, flush=True)
