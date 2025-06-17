from __future__ import absolute_import, division
import os
import os.path
import time
import sys
import numbers
import numpy as np
from collections import defaultdict
import pandas
import pandas as pd
from scipy.spatial.distance import euclidean
import sys

max_val = sys.maxsize
from numba import jit, float64, int32
from numba import njit
import matplotlib.pyplot as plt


def path_drawer(path, score_matrix):
    len_x = len(score_matrix)
    len_y = len(score_matrix[0])

    # 绘制距离矩阵热图
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 1, 1)
    plt.imshow(score_matrix, cmap='viridis', interpolation='nearest', origin='lower', vmin=0, vmax=100)
    plt.colorbar(label='Score Matrix Value')

    # 提取路径的 x 和 y 坐标
    path_x, path_y = zip(*path)

    # 在热图上绘制路径
    plt.plot(path_y, path_x, color='blue')

    plt.xlabel('Template Sequence')
    plt.ylabel('Test Sequence')

    plt.show()


@jit
def pose_sub(datax, datay, num_frames):
    lenx = len(datax)
    # print(lenx)
    sum_score = 0
    if lenx == 0:
        lenx = 10000
    point_score = 100.0 / lenx
    for i in range(lenx):  # 计算测试和模版一帧的每一个角度得分
        d_x = float(datax[i])
        d_y = float(datay[i])
        dif = abs(d_x - d_y)

        # 设定一个值的上下界
        t = 0.2
        if num_frames < 150:
            t = 0.1
        up_data = (1 + t) * d_x
        low_data = (1 - t) * d_x

        if low_data < d_y < up_data:
            # 不变
            point_score = 100.0 / lenx
        elif dif > 0.8 * d_y:  # 超过0.7倍数得1分
            point_score = 0.0
        else:  # 0.1到0.7之间的扣占比分
            if d_y == 0:
                p_c = 0
            else:
                p_c = (1 - dif / d_y)
            if p_c < 0:
                p_c = 0
            point_score = point_score * p_c
        sum_score = sum_score + point_score
    return sum_score


def expand_window(path, len_x, len_y, radius):
    """
    计算radius下的时间窗
    """
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius + 1)
                     for b in range(-radius, radius + 1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j
    return window


def reduce_by_half(x):
    """
    input list, make each two element together by half of sum of them
    :param x:
    :return:
    """
    x_reduce = []
    lens = len(x)
    for i in range(0, lens, 2):
        if (i + 1) >= lens:
            half = x[i]
        else:
            half = (x[i] + x[i + 1]) / 2
        x_reduce.append(half)
    return x_reduce


def find_trace(dis, score_dist, x, y, path_test, path_temp, num_frames, col_name):
    """
    :param dis:
    :param score_dist:
    :param x:
    :param y:
    :param path_test:
    :param path_temp:
    :param num_frames:
    :param col_name:
    :return:
    W: 回溯路径上的坐标列表。
    cnt: 回溯路径上坐标的数量。
    sum: 回溯路径上的累计得分。
    output_string: 包含错误分析结果的字符串。
    """
    cnt = 0
    m = np.size(dis, 0)
    n = np.size(dis, 1)
    sum = 0  # 用于累计路径上的得分
    W = []  # 用于存储回溯路径上的坐标
    i = m - 1
    j = n - 1
    output_string = ''
    column_names = col_name  #
    cnt_filter = 0
    frame_list = []  # 存储可能存在错误的帧对
    error_list = []  # 用于存储可能存在问题的帧号
    count_dict = {}  # 用于记录数字的出现次数
    frame_filter_list = []  # 筛选关键帧对
    while i > 0 or j > 0:
        sum += score_dist[i][j]
        # 某个位置的得分小于 60，则认为可能存在错误，将该位置加入 frame_list 并调用 pose_sub_error 函数进行进一步分析
        if score_dist[i][j] < 60:
            # print("frame {0} exist problem".format(i))
            # print("frame {0} in template".format(j))
            frame_list.append([i, j])
            # pose_sub_error(x[i - 1], y[j - 1], column_names, num_frames)

            error_list.append(int(i / 30))
        W.append((i, j))
        cnt += 1
        if i > 0 and j > 0:
            left_down = dis[i - 1, j - 1]
        else:
            left_down = max_val

        if i > 0:
            down = dis[i - 1, j]
        else:
            down = max_val

        if j > 0:
            left = dis[i, j - 1]
        else:
            left = max_val

        min_dis = min(left_down, down, left)
        if min_dis == left_down:
            i -= 1
            j -= 1
        elif min_dis == down:
            i -= 1
        else:
            j -= 1

    # W.append((0, 0))
    W = W[::-1]  # reverse W

    return W, cnt, sum, output_string


def dtw(x, y, num_frames, window=None):
    """
    输出参数：
    x: 第一个时间序列。
    y: 第二个时间序列。
    num_frames: 用于距离计算的参数。
    window: 可选参数，定义了搜索窗口，即允许的时间规整范围。如果未指定，则默认为整个序列范围

    返回参数：
    D: 存储距离矩阵的默认字典。（当前x坐标，当前y坐标）:（得分，上一个节点x坐标，上一个节点y坐标）
    path：回溯路径上的坐标:[(x,y),(x,y)]
    """
    len_x, len_y = len(x), len(y)  # 时间序列的长度
    # 定义一个二维数组，初始化为0，数组类型为float s是帧间得分矩阵
    s = np.full((len_x + 1, len_y + 1), 0, dtype=float)
    for i in range(1, len_x):
        s[i][0] = max_val
    for i in range(1, len_y):
        s[0][i] = max_val
    s[0][0] = 0
    # 确定搜索的范围
    # print('windows is :', window)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)

    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)  # (距离，x时间点来源，y时间点来源)
    # 若从左上角向右下角寻找最短路径过去的话
    for i, j in window:
        sc = pose_sub(x[i - 1], y[j - 1], num_frames)
        # print(i, j, sc)
        if sc == 0:
            sc = 1
        # print(i, j, sc)
        s[i][j] = sc
        dt = (1 / sc) * 100 - 1
        D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j), (D[i, j - 1][0] + dt, i, j - 1),
                      (D[i - 1, j - 1][0] + dt, i - 1, j - 1), key=lambda a: a[0])

    # 路径回溯，从终点坐标(len_x-1,len_y-1)开始
    path = []  # 存放路径坐标的列表

    i, j = len_x, len_y
    # print("len_x:{0},len_y:{1}".format(len_x, len_y))
    while not (i == j == 0):
        path.append((i - 1, j - 1))  # 首先将终点或者当前坐标加入path
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return D, path, s


def fastdtw(x, y, num_frames, col_name, radius=2):
    """
    x 测试时间序列
    y 模版时间序列
    num_frames 测试序列长度
    col_name 单帧上不同肢体之间的夹角，在该版本下是长度是27个
    """
    len_x, len_y = len(x), len(y)
    # print("len_x:{0},len_y:{1}".format(len_x, len_y))
    min_time_size = radius + 2  # 最小时间序列长度
    if len_x <= min_time_size or len_y <= min_time_size:
        return dtw(x, y, num_frames)
    x_shrinked = reduce_by_half(x)
    y_shrinked = reduce_by_half(y)
    D, path, s = fastdtw(x_shrinked, y_shrinked, num_frames, col_name, radius=radius)
    window = expand_window(path, len(x), len(y), radius)
    return dtw(x, y, num_frames, window)


def dtw_l(x, y, num_frames, col_name):
    """
    输出参数：
    x: 第一个时间序列。
    y: 第二个时间序列。
    num_frames: 用于距离计算的参数。
    window: 可选参数，定义了搜索窗口，即允许的时间规整范围。如果未指定，则默认为整个序列范围

    返回参数：
    D: 存储距离矩阵的默认字典。（当前x坐标，当前y坐标）:（得分，上一个节点x坐标，上一个节点y坐标）
    path：回溯路径上的坐标:[(x,y),(x,y)]
    """
    len_x, len_y = len(x), len(y)  # 时间序列的长度
    # 帧间得分矩阵 s
    s = np.full((len_x + 1, len_y + 1), 0, dtype=float)
    s[0][0] = 0
    # 帧间距离的矩阵 dist
    dist = np.full((len_x + 1, len_y + 1), 0, dtype=float)

    for i in range(1, len_x):
        s[i][0] = max_val
        dist[i][0] = dist[i - 1][0] + ((1 / pose_sub(x[i], y[0], num_frames)) * 100 - 1)
    for i in range(1, len_y):
        s[0][i] = max_val
        dist[0][i] = dist[0][i - 1] + ((1 / pose_sub(x[0], y[i], num_frames)) * 100 - 1)

    # 若从左上角向右下角寻找最短路径过去的话
    for i in range(1, len_x):
        for j in range(1, len_y):
            sc = pose_sub(x[i - 1], y[j - 1], num_frames)
            # print(i, j, sc)
            if sc == 0:
                sc = 1
            # print(i, j, sc)
            s[i][j] = sc
            dis = (1 / sc) * 100 - 1
            dist[i, j] = min(dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1]) + dis

    # 路径回溯，从终点坐标(len_x-1,len_y-1)开始
    path = []  # 存放路径坐标的列表
    i, j = len_x, len_y
    # print("len_x:{0},len_y:{1}".format(len_x, len_y))
    while True:
        if i > 0 and j > 0:
            path.append((i, j))  # 首先将终点或者当前坐标加入path
            m = min(dist[i - 1, j], dist[i, j - 1], dist[i - 1, j - 1])
            if m == dist[i - 1, j]:
                i -= 1
            elif m == dist[i, j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
        elif i == 0:
            path.append((i, j))
            j -= 1
        elif j == 0:
            path.append((i, j))
            i -= 1
        else:
            path.append((i, j))
            break
    path.reverse()
    return dist, path, s


if __name__ == '__main__':
    all_start = time.time()
    data_x = pandas.read_csv("E:/FJNU/project/PoseDetection/score_v2/data/data_x.csv")
    data_y = pandas.read_csv("E:/FJNU/project/PoseDetection/score_v2/data/data_y.csv")
    # path_test = "E:/FJNU/project/PoseDetection/score_v2/data/20231214_20231214065155A025.json"
    # path_temp = "E:/FJNU/project/PoseDetection/score_v2/data/3.1第一式双手托天理三焦_20231123001625A023.json"
    # result_str = ""
    # 关键点的提取
    # data_x, data_y, num_frames_test, num_frames_temp = action_keypoint(path_test, path_temp)

    data_x_numpy = data_x.to_numpy()
    data_y_numpy = data_y.to_numpy()
    co = ''
    num_frames = data_x.shape[0]
    print('test frames is', num_frames)
    al_start = time.time()
    # D, path, s = fastdtw(data_x_numpy, data_y_numpy, num_frames, co)
    D, path, s = dtw_l(data_x_numpy, data_y_numpy, num_frames, co)
    al_end = time.time()
    cnt = len(path)
    # print('路径上的点数len:', cnt)
    # print('\npath is :', path)
    score = 0
    for i, j in path:
        score += s[i][j]
    # print(score)
    end = time.time()
    print('score is :', (score / cnt), '\npath is:', path, "\n dtw algorithm cost_time:", (al_end - al_start),
          '\nall time cost:', (end - all_start))
    path_drawer(path, s)
    np.savetxt("../data/score_matrix.csv", s, delimiter=",", fmt='%f')
