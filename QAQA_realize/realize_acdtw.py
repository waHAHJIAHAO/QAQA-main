import os

import os.path
import numpy as np
import cv2
import time
import pandas as pd
import sys
import mediapipe as mp

from numba import jit, float64, int32
from numba import njit

max_val = sys.maxsize


@jit
def ed(m, n):
    m = m[1:-1]
    list1 = m.split(",")
    list1 = list(map(float, list1))

    n = n[1:-1]
    list2 = n.split(",")
    list2 = list(map(float, list2))

    return ((list1[0] - list2[0]) ** 2) + ((list1[1] - list2[1]) ** 2)


@jit
def print_angle(score, name):
    remark = "null"
    if name.startswith("angle"):
        # print("yes")
        if name == "angle01":
            if score < 0.5:
                remark = "右脖"
        if name == "angle02":
            if score < 0.5:
                remark = "右肩和躯干夹角"
        if name == "angle03":
            if score < 0.5:
                remark = "右肩"
        if name == "angle04":
            if score < 0.5:
                remark = "右肘"
        if name == "angle05":
            if score < 0.5:
                remark = "左肩膀"
        if name == "angle06":
            if score < 0.5:
                remark = "左肘"
        if name == "angle07":
            if score < 0.5:
                remark = "右臀"
        if name == "angle08":
            if score < 0.5:
                remark = "左臀"
        if name == "angle09":
            if score < 0.5:
                remark = "右膝"
        if name == "angle10":
            if score < 0.5:
                remark = "右脚踝"
        if name == "angle11":
            if score < 0.5:
                remark = "左膝"
        if name == "angle12":
            if score < 0.5:
                remark = "左脚踝"
        # if remark != "null":  # 检查remark是否为空字符串
        #     print(remark)
    return remark


# @jit
def pose_sub_error(datax, datay, name, num_frames):
    lenx = len(datax)
    # print(lenx)
    sum_score = 0
    point_score = 100.0 / lenx
    name_list = []
    for i in range(lenx):
        d_x = float(datax[i])
        d_y = float(datay[i])
        dif = abs(d_x - d_y)
        # print(d_x)
        # print(d_y)

        # 设定一个值的上下界
        t = 0.2
        if num_frames < 150:
            t = 0.1
        up_data = (1 + t) * d_x
        low_data = (1 - t) * d_x

        if low_data < d_y < up_data:
            # 不变
            point_score = 100.0 / lenx
        elif dif > 0.8 * d_x:  # 超过0.7倍数得1分
            point_score = 0.0
        else:  # 0.1到0.7之间的扣占比分
            if d_y == 0:
                p_c = 0
            else:
                p_c = (1 - dif / d_y)
            if p_c < 0:
                p_c = 0
            point_score = point_score * p_c

        angle_name = print_angle(point_score, name[i])
        if angle_name != "null":
            name_list.append(angle_name)
        sum_score = sum_score + point_score
    return name_list


# pose_sub比较两个时间序列中对应位置的数据点，计算它们之间的相似度得分。得分越高表示两个序列在该位置越相似
@jit()
def pose_sub(datax, datay, num_frames):
    lenx = len(datax)
    # print(f'lenx in pose_sub :{lenx}')
    sum_score = 0
    if lenx == 0:
        lenx = 10000
    point_score = 100.0 / lenx
    for i in range(lenx):
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
            # 认为是可接受的误差，分数保持不变
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


@jit
def cnt_out(dist, path, average):
    cnt = 0
    for p in path:
        if dist[p[0]][p[1]] < average - 15:
            cnt = cnt + 1
            print(dist[p[0]][p[1]])
    # print(cnt)
    return cnt


@jit
def pose_score(dis):
    sim = 1.0 / (1.0 + dis)
    score = sim * 100
    return score


# 获取错误动作对应的图像路径
def path_error_image(f_x, f_y, test_video, temp_video, folder_path):
    print("enter path_error_image-----------")
    path_student = folder_path + "/" + str(f_x) + ".png"
    path_teacher = folder_path + "/t_" + str(f_y) + ".png"

    for i in range(2):
        if i == 0:
            video_path = test_video
            target = f_x
            image_path = path_student
        if i == 1:
            video_path = temp_video
            target = f_y
            image_path = path_teacher
        # 打开视频文件
        # video_path = "E:/01_tiyu/02 对接/test/template_1.mp4"
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("无法打开视频文件")
            exit()

        # 读取指定帧

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # print("sssssssssssssssss")
            frame_count += 1

            if frame_count == target:
                # 保存图像
                # image_path = path_student
                cv2.imwrite(image_path, frame)

                # print(f"第{target}帧已保存为{image_path}")
                break

    # 释放视频文件和OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()
    return path_student, path_teacher


def path_point_image(image_path):
    # 加载MediaPipe的Pose模型
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # 读取图像
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # 对图像进行骨骼点识别
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output_image_path = "0"
    # 绘制骨骼点
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 输出带有骨骼点的图片
        output_image_path = image_path[:-4] + "_point" + ".png"
        cv2.imwrite(output_image_path, annotated_image)
        # print(f"带有骨骼点的图片已保存为{output_image_path}")
    else:
        print("无法识别骨骼点")

    # 释放资源
    pose.close()
    return output_image_path


# 回溯最佳路径
# @jit
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
    print("dis matrix is ",dis)
    cnt = 0
    m = np.size(dis, 0)
    n = np.size(dis, 1)
    sum = 0  # 用于累计路径上的得分
    W = []  # 用于存储回溯路径上的坐标
    i = m - 1
    j = n - 1
    column_names = col_name  #
    cnt_filter = 0
    frame_list = []  # 存储可能存在错误的帧对
    error_list = []  # 用于存储可能存在问题的帧号
    count_dict = {}  # 用于记录数字的出现次数
    frame_filter_list = []  # 筛选关键帧对
    # print(score_dist)
    while i > 0 or j > 0:
        sum += score_dist[i][j]
        print(f"score_dist[{i}][{j}]", score_dist[i][j])
        # 某个位置的得分小于 60，则认为可能存在错误，将该位置加入 frame_list 并调用 pose_sub_error 函数进行进一步分析
        if score_dist[i][j] < 60:
            # print("frame {0} exist problem".format(i))
            # print("frame {0} in template".format(j))
            frame_list.append([i, j])
            pose_sub_error(x[i - 1], y[j - 1], column_names, num_frames)

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

    # 倒转frame_list
    frame_list = frame_list[::-1]
    # 统计数字出现次数
    for num in error_list:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1

    # 保留出现次数大于3次的数字
    cishu = 4
    if num_frames < 150:
        cishu = 1
    filtered_list = [num for num in error_list if count_dict[num] >= cishu]  # 错误次数大于4 则记录下来
    filtered_list = list(set(filtered_list))  # 去除重复的数字

    filtered_list.sort()
    miao_list = filtered_list
    # print("filtered_list = {0}".format(filtered_list))  # 打印筛选后的列表
    filtered_list = [x * 30 for x in filtered_list]
    # print("filtered_list1111 = {0}".format(filtered_list))  # 打印筛选后的列表
    # print(frame_list)
    for element in frame_list:
        if cnt_filter == len(filtered_list):
            break
        if filtered_list[cnt_filter] <= element[0] <= filtered_list[cnt_filter] + 30:
            # print(element[0])
            # print(filtered_list[cnt_filter] )
            frame_filter_list.append(element)
            cnt_filter += 1
        elif element[0] > filtered_list[cnt_filter] + 30:
            cnt_filter += 1
    # print(frame_filter_list)  # 打印筛选后的列表
    # filtered_list = list(set(filtered_list))
    # print("--------------------------")
    # print(miao_list)

    take_num = 0
    # 获取当前时间戳
    timestamp = int(time.time_ns())

    # 指定存储错误动作图片信息的文件路径
    path = './img'  # 替换为你的目标路径
    if not os.path.exists(path):
        os.mkdir(path)
    # 创建文件夹
    folder_name = str(timestamp)
    folder_path = os.path.join(path, folder_name)

    if len(miao_list) >= 1:
        # 检查文件夹是否已经存在
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        else:
            print(f"文件夹 '{folder_name}' 已存在，无需创建")
    output_string = ""
    # print("len(miao_list) = ")
    # print(len(miao_list))
    for ele in miao_list:

        if take_num >= 3:
            break
        index_x = frame_filter_list[take_num][0] - 1

        index_y = frame_filter_list[take_num][1] - 1
        if index_x <= 0:
            index_x = 1
        if index_y <= 0:
            index_y = 1
        error_name = pose_sub_error(x[index_x], y[index_y],
                                    column_names, num_frames)
        # print("------------------------------------------")
        # print("error_name = {0}".format(error_name))
        print("len,frame_filter_list = {0}".format(len(frame_filter_list)))
        error_name_str = ','.join(error_name)
        print("====================================================")
        print(frame_filter_list[take_num][0])
        print("take_Num = {0}".format(take_num))

        path_student, path_teacher = path_error_image(index_x, index_y,
                                                      path_test, path_temp, folder_path)
        if path_student is not None and path_teacher is not None:
            # 调用模型绘画错误动作的人体关键点图片，这个可以换成mmpose中快一点的模型
            path_student = path_point_image(path_student)
            path_teacher = path_point_image(path_teacher)
            if path_teacher == "0" or path_student == "0":
                take_num += 1
                continue
            if take_num < len(miao_list) - 1 and take_num < 2:
                output_string += "在第{0}秒可能存在{1}的问题|学生图片路径{2}|老师图片路径{3}*".format(ele,
                                                                                error_name_str,
                                                                                path_student,
                                                                                path_teacher)
                # print("在第{0}秒可能存在{1}的问题|学生图片路径{2}|老师图片路径{3}*".format(ele, error_name_str, path_student, path_teacher))
            else:
                output_string += "在第{0}秒可能存在{1}的问题|学生图片路径{2}|老师图片路径{3}".format(ele,
                                                                               error_name_str,
                                                                               path_student,
                                                                               path_teacher)
        take_num += 1
    return W, cnt, sum, output_string


# @jit
def find_trace_order(dis, score_dist):
    cnt = 0
    m = np.size(dis, 0)
    n = np.size(dis, 1)
    sum = 0
    W = []
    i = 1
    j = 1
    while i < m or j < n:
        sum += score_dist[i][j]
        W.append((i, j))
        cnt += 1
        # print(i,j)
        if i < m - 1 and j < n - 1:
            left_down = dis[i + 1, j + 1]
        else:
            left_down = max_val

        if i < m - 1:
            down = dis[i + 1, j]
        else:
            down = max_val

        if j < n - 1:
            left = dis[i, j + 1]
        else:
            left = max_val

        min_dis = min(left_down, down, left)
        # print(left_down, down, left)
        if min_dis == left_down:
            i += 1
            j += 1
            # print('left_down')
        elif min_dis == down:
            i += 1
            # print('down')
        else:
            j += 1
    W = W[::-1]
    # print(W)
    return W, cnt, sum


def realize_dtw(x, y, path_test, path_temp, num_frames, col_name):
    lam = 0.25
    # lam: float64 = 0.25
    # maxn = np.inf
    Num_col = x.shape[1]
    # 需要比对的时间序列长度
    lenx = x.shape[0] + 1
    leny = y.shape[0] + 1
    # print(y)
    # print('num_col', Num_col)
    # print('lenx:', lenx)
    # print('leny:', leny)
    _mn = 2 * max(lenx, leny) / (lenx + leny)
    # Q和C矩阵做约束
    T_Q = np.zeros((lenx, leny), dtype="int")
    T_C = np.zeros((lenx, leny), dtype="int")
    # 存储最终的得分矩阵
    dist = np.full((lenx, leny), 0, dtype=float)
    # warping_dis用于存储规整后的成本矩阵
    warping_dis = np.ones((lenx, leny), dtype="float64") * max_val
    warping_dis[0][0] = 0
    # 定义一个二维数组，初始化为0，数组类型为float
    for i in range(1, lenx):
        dist[i][0] = max_val
    for i in range(1, leny):
        dist[0][i] = max_val
    dist[0][0] = 0
    _Copy = dist.copy()
    # 算法预处理，计算初步的得分矩阵,这里的O(n^2)可以简化到下面的warping_dis的双重for循环中
    for i in range(1, lenx):
        for j in range(1, leny):
            med = pose_sub(x[i - 1], y[j - 1], num_frames)
            _Copy[i][j] = med
            if med == 0:
                med = 1
            dist[i][j] = (1 / med) * 100 - 1
    '''
    rx = int(lam * lenx)
    ry = int(lam * leny)
    k = (leny * 1.0) / (lenx * 1.0)
    end_rx = lenx - rx
    # 全局约束
    for i in range(end_rx):
        start = int(k * i + ry)
        for j in range(start, leny):
            dist[i][j] = max_val

    for i in range(rx, lenx):
        end = int(k * i - ry)
        if end < 0:
            end = 0
        for j in range(end):
            dist[i][j] = max_val
    '''
    print("算法预处理完成")
    # ac-dtw算法执行
    for i in range(1, lenx):
        for j in range(1, leny):
            # if i == 0 and j == 0:
            #     warping_dis[0, 0] = dist[0, 0]
            #     continue
            if i > 0 and j > 0:
                left_down = warping_dis[i - 1, j - 1]
            else:
                left_down = max_val

            if i > 0:
                down = warping_dis[i - 1, j] + _mn * (T_C[i - 1][j] + T_Q[i - 1][j]) * dist[i, j]
            else:
                down = max_val

            if j > 0:
                left = warping_dis[i, j - 1] + _mn * (T_C[i][j - 1] + T_Q[i][j - 1]) * dist[i, j]
            else:
                left = max_val
            min_tmp = min(left_down, down, left)
            if min_tmp >= max_val:
                continue
            warping_dis[i, j] = dist[i, j] + min_tmp
            if min_tmp == left_down:
                T_Q[i][j] = 1
                T_C[i][j] = 1
            elif min_tmp == down:
                T_Q[i][j] = 1
                T_C[i][j] = T_C[i - 1][j] + 1
            elif min_tmp == left:
                T_Q[i][j] = T_Q[i][j - 1] + 1
                T_C[i][j] = 1
    print("算法处理完成，寻找路径中")
    # 回溯最优路径 路径上的点数目为：w_cnt,距离总和：w_sum
    W, w_cnt, w_sum, result_str = find_trace(warping_dis, _Copy, x, y, path_test, path_temp, num_frames, col_name)
    result_str = "寻找路径中" + result_str
    path1 = []
    path2 = []
    a = lenx - 1
    b = leny - 1
    cnt = 0

    dist1 = dist[1:, 1:]
    # print('cnt:', cnt)
    print("w_cnt:", w_cnt, "w_sum:", w_sum)
    print('\npath is', W)
    w_score = w_sum / w_cnt  # 总分数/点的个数=每个动作的平均得分
    return w_score, result_str
