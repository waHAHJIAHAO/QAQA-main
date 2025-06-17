import numpy as np
import time
import matplotlib.pyplot as plt
import math
import pandas as pd
import copy

from realize.realize_data import list_sub, dot_product_angle


def calc_angle_vector(d_iloc, v_a, v_b, v_c, v_d):
    vertex_b = d_iloc[v_b]
    vertex_a = d_iloc[v_a]
    vertex_c = d_iloc[v_c]
    vertex_d = d_iloc[v_d]

    vec_ba = list_sub(vertex_a, vertex_b)
    vec_dc = list_sub(vertex_c, vertex_d)
    return dot_product_angle(np.array(vec_ba), np.array(vec_dc))


def dot_product_angle_sin(v1, v2):
    # print(v1,v2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arcsin = np.arcsin(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # 弧度

        angle = np.degrees(arcsin)
        # 度数
        return arcsin


def arm_leg_angle(data):
    data_row = data.shape[0]
    data_col = data.shape[1]
    df = pd.DataFrame(None, columns=['arm_leg01'])
    df_angle01 = pd.DataFrame(None, columns=['angle'])
    df_angle02 = pd.DataFrame(None, columns=['angle'])
    df_angle03 = pd.DataFrame(None, columns=['angle'])
    df_angle04 = pd.DataFrame(None, columns=['angle'])

    for i in range(data_row):
        df_angle01.loc[i] = (calc_angle_vector(data.iloc[i], 'RWrist', 'RElbow', 'RAnkle', 'RKnee'))
        df_angle02.loc[i] = (calc_angle_vector(data.iloc[i], 'LWrist', 'LElbow', 'LAnkle', 'LKnee'))
        df_angle03.loc[i] = (calc_angle_vector(data.iloc[i], 'RWrist', 'RElbow', 'RKnee', 'RHip'))
        df_angle04.loc[i] = (calc_angle_vector(data.iloc[i], 'LWrist', 'LElbow', 'LKnee', 'LHip'))
    df['arm_leg01'] = df_angle01['angle']
    df['arm_leg02'] = df_angle02
    df['arm_leg03'] = df_angle03
    df['arm_leg04'] = df_angle04

    return df


def Shoulder_Hip_angle(data):
    data_row = data.shape[0]
    data_col = data.shape[1]
    df = pd.DataFrame(None, columns=['Hip_angle01'])
    df_angle01 = pd.DataFrame(None, columns=['angle'])
    df_angle02 = pd.DataFrame(None, columns=['angle'])
    df_angle03 = pd.DataFrame(None, columns=['angle'])
    df_angle04 = pd.DataFrame(None, columns=['angle'])

    df_dis01 = pd.DataFrame(None, columns=['open01'])
    df_dis02 = pd.DataFrame(None, columns=['open02'])

    # arm_leg_angle
    df_angle05 = pd.DataFrame(None, columns=['angle'])
    df_angle06 = pd.DataFrame(None, columns=['angle'])
    df_angle07 = pd.DataFrame(None, columns=['angle'])
    df_angle08 = pd.DataFrame(None, columns=['angle'])

    list_a = [0, 0, 1]
    list_b = [1, 0, 0]
    for i in range(data_row):

        # arm_leg_angle
        df_angle05.loc[i] = (calc_angle_vector(data.iloc[i], 'RWrist', 'RElbow', 'RAnkle', 'RKnee'))
        df_angle06.loc[i] = (calc_angle_vector(data.iloc[i], 'LWrist', 'LElbow', 'LAnkle', 'LKnee'))
        df_angle07.loc[i] = (calc_angle_vector(data.iloc[i], 'RWrist', 'RElbow', 'RKnee', 'RHip'))
        df_angle08.loc[i] = (calc_angle_vector(data.iloc[i], 'LWrist', 'LElbow', 'LKnee', 'LHip'))

        # Shoulder_Hip_angle
        v_a = copy.deepcopy(data.iloc[i]['LHip'])
        v_b = copy.deepcopy(data.iloc[i]['RHip'])
        v_c = copy.deepcopy(data.iloc[i]['LShoulder'])
        v_d = copy.deepcopy(data.iloc[i]['RShoulder'])

        # 向量差
        # vec_ba = list_sub(v_a[:-1], v_b[:-1])
        # vec_dc = list_sub(v_c[:-1], v_d[:-1])
        vec_ba = list_sub(v_a, v_b)
        vec_dc = list_sub(v_c, v_d)
        df_angle01.loc[i] = dot_product_angle_sin(np.array(vec_ba), np.array(list_a)) + (math.pi / 2)
        df_angle02.loc[i] = dot_product_angle_sin(np.array(vec_ba), np.array(list_b)) + (math.pi / 2)
        df_angle03.loc[i] = dot_product_angle_sin(np.array(vec_dc), np.array(list_a)) + (math.pi / 2)
        df_angle04.loc[i] = dot_product_angle_sin(np.array(vec_dc), np.array(list_b)) + (math.pi / 2)

        # open_arm
        v_a = copy.deepcopy(data.iloc[i]['LAnkle'])
        v_b = copy.deepcopy(data.iloc[i]['RAnkle'])

        v_lw = copy.deepcopy(data.iloc[i]['LWrist'])
        v_rw = copy.deepcopy(data.iloc[i]['RWrist'])

        v_c = copy.deepcopy(data.iloc[i]['LShoulder'])
        v_d = copy.deepcopy(data.iloc[i]['RShoulder'])

        if distance(v_a, v_b) > (0.5 * distance(v_c, v_d)):
            df_dis01.loc[i] = 1.5 * math.pi
        else:
            df_dis01.loc[i] = 0.5 * math.pi

        if distance(v_lw, v_rw) > (1.5 * distance(v_c, v_d)):
            df_dis02.loc[i] = 1.5 * math.pi
        else:
            df_dis02.loc[i] = 0.5 * math.pi

    # Shoulder_Hip_angle 上身与下身之间的角度
    df['Hip_angle01'] = df_angle01['angle']
    df['Hip_angle02'] = df_angle02
    df['Shoulder_angle01'] = df_angle03
    df['Shoulder_angle02'] = df_angle04

    # open_arm 两腿、两手之间的开放程度
    df['open01'] = df_dis01['open01']
    df['open02'] = df_dis02

    # arm_leg_angle 手臂与腿之间的角度
    df['arm_leg01'] = df_angle05['angle']
    df['arm_leg02'] = df_angle06
    df['arm_leg03'] = df_angle07
    df['arm_leg04'] = df_angle08

    return df


def distance(v_a, v_b):
    return ((v_a[0] - v_b[0]) ** 2 + (v_b[1] - v_b[1]) ** 2 + (v_a[2] - v_b[2]) ** 2) ** 0.5


def open_arm(data):
    data_row = data.shape[0]
    data_col = data.shape[1]
    df = pd.DataFrame(None, columns=['open01'])
    df_dis01 = pd.DataFrame(None, columns=['open01'])
    df_dis02 = pd.DataFrame(None, columns=['open02'])

    for i in range(data_row):
        v_a = copy.deepcopy(data.iloc[i]['LAnkle'])
        v_b = copy.deepcopy(data.iloc[i]['RAnkle'])

        v_lw = copy.deepcopy(data.iloc[i]['LWrist'])
        v_rw = copy.deepcopy(data.iloc[i]['RWrist'])

        v_c = copy.deepcopy(data.iloc[i]['LShoulder'])
        v_d = copy.deepcopy(data.iloc[i]['RShoulder'])

        if distance(v_a, v_b) > (0.5 * distance(v_c, v_d)):
            df_dis01.loc[i] = 1.5 * math.pi
        else:
            df_dis01.loc[i] = 0.5 * math.pi

        if distance(v_lw, v_rw) > (1.5 * distance(v_c, v_d)):
            df_dis02.loc[i] = 1.5 * math.pi
        else:
            df_dis02.loc[i] = 0.5 * math.pi

    df['open01'] = df_dis01['open01']
    df['open02'] = df_dis02

    return df
