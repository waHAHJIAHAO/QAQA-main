import numpy as np
import time
import math
import pandas as pd
import copy


def to_1(mold):
    """
    m = copy.deepcopy(mold)
    m = m[1:-1]
    print("-----------------------------------.{0}".format(type(m)))
    list1 = m.split(", ")
    list1 = list(map(float, list1))
    # 只取前两个元素
    """
    return mold[:3]


def to_(arr_data):
    # print('ssssss')
    # to_res = np.array()
    to_res = list()
    last = list()
    lenarr = len(arr_data)
    for i in range(lenarr):
        # np.append(to_res,to_1(arr_data[i]))
        if len(arr_data[i]) > 2:
            # print("arr_data[i] = {0}".format(arr_data[i]))
            last = to_1(arr_data[i])
            to_res.append(last)
        else:
            if last:
                to_res.append(last)
            else:
                last = to_1(arr_data[i + 1])
                to_res.append(last)
    return to_res


def list_add_div(list1, list2):
    res = list(map(lambda x: (x[0] + x[1]) / 2, zip(list1, list2)))

    return res


def Coordinate_Neck(data):
    """
    输入的数据中计算出中间臀部 (MidHip) 和颈部 (Neck) 的坐标，
    并对所有关键点坐标进行平移，使得颈部的关键点坐标变为 (0, 0, 0)
    """
    data_row = data.shape[0]
    data_col = data.shape[1]
    # print("data_col = {0}".format(data_col))
    new_data = pd.DataFrame()
    # 初始化
    for i in range(data_col):
        array = data[data.columns[i]]
        new_data[data.columns[i]] = array

    # df_MidHip = pd.DataFrame({'MidHip': [[1.1, 2.1, 1.1, 1.1], [3.1, 4.1, 5.1, 1.1]]})
    # df_Neck = pd.DataFrame({'Neck': [[1.1, 2.1, 3.1, 1.1], [3.1, 4.1, 5.1, 1.1]]})
    #
    # for i in range(data_row):
    #     array = (new_data.iloc[i])
    #     # print("array = {0}".format(array))
    #     list_RHip = copy.deepcopy(array['RHip'])
    #     list_LHip = copy.deepcopy(array['LHip'])
    #     list_MidHip = list_add_div(list_LHip, list_RHip)
    #
    #     list_RShoulder = copy.deepcopy(array['RShoulder'])
    #     list_LShoulder = copy.deepcopy(array['LShoulder'])
    #     list_Neck = list_add_div(list_RShoulder, list_LShoulder)
    #
    #     df_MidHip.loc[i] = [list_MidHip]
    #     df_Neck.loc[i] = [list_Neck]
    # """添加中间臀部和颈部坐标到新 DataFrame"""
    # new_data['MidHip'] = df_MidHip
    # # print(df_MidHip)
    # new_data['Neck'] = df_Neck
    # #print(new_data)
    """ 平移关键点坐标,以neck为root，每个三维坐标都减去neck的x,y,z，
    注：“这边的颈部已经不是原本mediapipe中
       左右肩膀计算取平均计算后的值，是mmpose中提取出的thorax值，
       在图像上看MP和MMPOSE这两个区域关键点的位置相近“
    """
    for i in range(new_data.shape[0]):
        array_r = new_data.iloc[i]
        # print(array_r)
        tmp = array_r['Neck']
        tmp_x = tmp[0]
        tmp_y = tmp[1]
        tmp_z = tmp[2]
        for j in range(new_data.shape[1]):
            # print(array_r[j])
            # print(tmp_x)
            array_r.iloc[j][0] = array_r.iloc[j][0] - tmp_x
            array_r.iloc[j][1] = array_r.iloc[j][1] - tmp_y
            array_r.iloc[j][2] = array_r.iloc[j][2] - tmp_z
        new_data.iloc[i] = array_r
        # time.sleep(2)
    return new_data


# 算向量夹角
def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # 弧度

        angle = np.degrees(arccos)
        # 度数
        return arccos


def center_gravity(data):
    dict = {
        'Nose': 0.0706, 'Neck': 0.2391, 'RWrist': 0.0096,
        'LWrist': 0.0096, 'RShoulder': 0.0178, 'LShoulder': 0.0178,
        'RElbow': 0.029, 'LElbow': 0.029, 'MidHip': 0.1879,
        'RHip': 0.06485, 'LHip': 0.06485, 'RKnee': 0.0815,
        'LKnee': 0.0815, 'RAnkle': 0.03215, 'LAnkle': 0.03215,
        'RHeel': 0.0079, 'LHeel': 0.0079,
        'RHand': 0.0090, 'LHand': 0.0090
    }

    data_row = data.shape[0]
    data_col = data.shape[1]

    new_data = data
    # new_data.drop(columns=['LEye', 'REye', 'LEar', 'REar', 'RBigToe', 'LBigToe', 'RHSmallToe', 'LHSmallToe'],
    #              inplace=True)

    # 写那个手重心 （3-9）
    df_Rhand = pd.DataFrame({'Rhand': [[1.1, 2.1, 1.1], [3.1, 4.1, 5.1]]})
    df_Lhand = pd.DataFrame({'Lhand': [[1.1, 2.1, 3.1], [3.1, 4.1, 5.1]]})
    q = 1 / 4
    # 应该是1/4而不是1/9
    for i in range(data_row):
        array = new_data.iloc[i]
        list_RPinky = copy.deepcopy(array['RPinky'])
        list_RIndex = copy.deepcopy(array['RIndex'])
        list_RWrist = copy.deepcopy(array['RWrist'])
        list_RMid = list_add_div(list_RPinky, list_RIndex)
        list_Rhand = list_add_div(list_RMid, list_RWrist)

        list_LPinky = copy.deepcopy(array['LPinky'])
        list_LIndex = copy.deepcopy(array['LIndex'])
        list_LWrist = copy.deepcopy(array['LWrist'])
        list_LMid = list_add_div(list_LPinky, list_LIndex)
        list_Lhand = list_add_div(list_LMid, list_LWrist)

        df_Rhand.loc[i] = [list_Rhand]
        df_Lhand.loc[i] = [list_Lhand]

    # print((df_rhand))
    new_data['RHand'] = df_Rhand
    # print(new_data)
    new_data['LHand'] = df_Lhand
    # time.sleep(2)

    new_data.drop(columns=['LPinky', 'RPinky', 'LIndex', 'RIndex',
                           'LThumb', 'RThumb', 'LBigToe', 'RBigToe'], inplace=True)

    df_gravity = pd.DataFrame({'gravity': [[1.1, 2.1, 2.1], [3.1, 4.1, 4.1]]})
    df_angle = pd.DataFrame({'angle': [1.1, 2.1, 3.1]})
    for i in range(new_data.shape[0]):
        x_gravity = 0
        y_gravity = 0
        z_gravity = 0
        for j in range(new_data.shape[1]):
            name_col = new_data.columns[j]
            # print('name:', name_col)
            k = dict[name_col]
            x_gravity = x_gravity + k * new_data.iloc[i][name_col][0]
            y_gravity = y_gravity + k * new_data.iloc[i][name_col][1]
            z_gravity = z_gravity + k * new_data.iloc[i][name_col][2]
        list_gravity = list([x_gravity, y_gravity, z_gravity])
        x_MidHip = new_data.iloc[i]['MidHip'][0]
        y_MidHip = new_data.iloc[i]['MidHip'][1]
        z_MidHip = new_data.iloc[i]['MidHip'][2]

        # 三维夹角！！！！
        tmplist_gravity = list([x_gravity, y_gravity, z_gravity])
        tmplist_MidHip = list([x_MidHip, y_MidHip, z_MidHip])
        # deno = np.sqrt(((x_gravity ** 2) + (y_gravity ** 2)) * ((x_MidHip ** 2) + (y_MidHip ** 2)))
        angle = dot_product_angle(np.array(tmplist_gravity), np.array(tmplist_MidHip))
        # df_gravity.append(list_gravity)
        # df_angle = df_angle.append(angle)
        df_gravity.loc[i] = [list_gravity]
        df_angle.loc[i] = angle

    # print(df_gravity)
    # print(df_angle)
    new_data['gravity'] = df_gravity
    new_data['gravity_angle'] = df_angle
    # print(df_gravity)
    return new_data['gravity_angle']


def list_sub(list1, list2):
    res = list(map(lambda x: x[0] - x[1], zip(list1, list2)))

    return res


def list_mul(list1, list2):
    res = list(map(lambda x: x[0] * x[1], zip(list1, list2)))
    return res


def list_mul_add(list1, list2):
    tmp = list_mul(list1, list2)
    res = 0
    for i in range(len(tmp)):
        res = res + tmp[i]
    return res


def calc_angle(d_iloc, v_b, v_a, v_c):
    # print(d_iloc)
    vertex_b = d_iloc[v_b]
    vertex_a = d_iloc[v_a]
    vertex_c = d_iloc[v_c]
    vec_ba = list_sub(vertex_a, vertex_b)
    vec_bc = list_sub(vertex_c, vertex_b)
    angle = dot_product_angle(np.array(vec_ba), np.array(vec_bc))
    return angle


def calc_angle_limb(d_iloc, v_a, v_b, v_c, v_d):
    arr_b = np.array(d_iloc[v_b])
    arr_a = np.array(d_iloc[v_a])
    arr_c = np.array(d_iloc[v_c])
    arr_d = np.array(d_iloc[v_d])
    arr_spine = np.array(d_iloc['MidHip'])

    p = (arr_a + arr_b + arr_c + arr_d) / 4

    angle = dot_product_angle(p, arr_spine)
    return angle


# 肢体夹角求解
def limb_angle(data):
    data_row = data.shape[0]
    data_col = data.shape[1]
    df = pd.DataFrame(None, columns=['angle01'])
    df_angle01 = pd.DataFrame(None, columns=['angle01'])
    df_angle02 = pd.DataFrame(None, columns=['angle02'])
    df_angle03 = pd.DataFrame(None, columns=['angle03'])
    df_angle04 = pd.DataFrame(None, columns=['angle04'])
    df_angle05 = pd.DataFrame(None, columns=['angle05'])
    df_angle06 = pd.DataFrame(None, columns=['angle06'])
    df_angle07 = pd.DataFrame(None, columns=['angle07'])
    df_angle08 = pd.DataFrame(None, columns=['angle08'])
    df_angle09 = pd.DataFrame(None, columns=['angle09'])
    df_angle10 = pd.DataFrame(None, columns=['angle10'])
    df_angle11 = pd.DataFrame(None, columns=['angle11'])
    df_angle12 = pd.DataFrame(None, columns=['angle12'])
    df_angle13 = pd.DataFrame(None, columns=['angle13'])

    df_limb_angle01 = pd.DataFrame(None, columns=['limb_angle01'])
    df_limb_angle02 = pd.DataFrame(None, columns=['limb_angle02'])
    df_limb_angle03 = pd.DataFrame(None, columns=['limb_angle03'])
    df_limb_angle04 = pd.DataFrame(None, columns=['limb_angle04'])

    for i in range(data_row):
        # 以B为顶点
        # 计算不同肢体关节的角度
        df_angle01.loc[i] = (calc_angle(data.iloc[i], 'Neck', 'Nose', 'RShoulder'))
        df_angle02.loc[i] = (calc_angle(data.iloc[i], 'Neck', 'MidHip', 'RShoulder'))
        df_angle03.loc[i] = (calc_angle(data.iloc[i], 'RShoulder', 'Neck', 'RElbow'))
        df_angle04.loc[i] = (calc_angle(data.iloc[i], 'RElbow', 'RShoulder', 'RWrist'))
        df_angle05.loc[i] = (calc_angle(data.iloc[i], 'LShoulder', 'Neck', 'LElbow'))
        df_angle06.loc[i] = (calc_angle(data.iloc[i], 'LElbow', 'LShoulder', 'LWrist'))
        df_angle07.loc[i] = (calc_angle(data.iloc[i], 'MidHip', 'Neck', 'RHip'))
        df_angle08.loc[i] = (calc_angle(data.iloc[i], 'RHip', 'MidHip', 'RKnee'))
        df_angle09.loc[i] = (calc_angle(data.iloc[i], 'RKnee', 'RHip', 'RAnkle'))
        # df_angle10.loc[i] = (calc_angle(data.iloc[i], 'RAnkle', 'RKnee', 'RBigToe'))
        df_angle11.loc[i] = (calc_angle(data.iloc[i], 'LHip', 'MidHip', 'LKnee'))
        df_angle12.loc[i] = (calc_angle(data.iloc[i], 'LKnee', 'LHip', 'LAnkle'))
        # df_angle13.loc[i] = (calc_angle(data.iloc[i], 'LAnkle', 'LKnee', 'LBigToe'))
        # 角度计算肢体的整体
        df_limb_angle01.loc[i] = (calc_angle_limb(data.iloc[i], 'Neck', 'RShoulder', 'RElbow', 'RWrist'))
        df_limb_angle02.loc[i] = (calc_angle_limb(data.iloc[i], 'Neck', 'LShoulder', 'LElbow', 'LWrist'))
        df_limb_angle03.loc[i] = (calc_angle_limb(data.iloc[i], 'MidHip', 'RHip', 'RKnee', 'RAnkle'))
        df_limb_angle04.loc[i] = (calc_angle_limb(data.iloc[i], 'MidHip', 'LHip', 'LKnee', 'LAnkle'))

    # print(df_angle01)
    # print(df_limb_angle01)
    # 合并角度信息
    df['angle01'] = df_angle01['angle01']
    df['angle02'] = df_angle02
    df['angle03'] = df_angle03
    df['angle04'] = df_angle04
    df['angle05'] = df_angle05
    df['angle06'] = df_angle06
    df['angle07'] = df_angle07
    df['angle08'] = df_angle08
    df['angle09'] = df_angle09
    # df['angle10'] = df_angle10
    df['angle11'] = df_angle11
    df['angle12'] = df_angle12
    # df['angle13'] = df_angle13

    df['limb_angle01'] = df_limb_angle01
    df['limb_angle02'] = df_limb_angle02
    df['limb_angle03'] = df_limb_angle03
    df['limb_angle04'] = df_limb_angle04

    return df


# 计算身体朝向
def body_orient(data):
    df = pd.DataFrame()
    data_row = data.shape[0]
    df_orient01 = pd.DataFrame(None, columns=['orient'])
    df_orient02 = pd.DataFrame(None, columns=['orient'])
    df_orient03 = pd.DataFrame(None, columns=['orient'])

    for i in range(data_row):
        '''
        s.append(i[5][0]-i[2][0])   #5:左肩，2：右肩
        h.append(i[12][0]-i[9][0])  #12:左髋关节，9：右髋关节
        f.append((i[22]-i[24])+(i[19]-i[21]))   #22、24：右大脚趾、右脚跟，19、21：左大脚趾、左脚跟
        '''
        tmp = data.iloc[i]
        o1, o2, o3 = 0, 0, 0
        # 上肢体朝向 上身朝向角度根据左右肩膀的位置确定
        if tmp['LShoulder'][0] - tmp['RShoulder'][0] > 0:
            o1 = 1.5 * math.pi
        else:
            o1 = 0.5 * math.pi
        # 下肢体朝向 下身朝向角度根据左右髋关节的位置确定
        if tmp['LHip'][0] - tmp['RHip'][0] > 0:
            o2 = 1.5 * math.pi
        else:
            o2 = 0.5 * math.pi
        # 人体左右朝向,是否要+pi 整体朝向角度根据左右大脚趾和脚跟的位置计算得出
        # f0 = [
        #     (tmp['RBigToe'][0] - tmp['RHeel'][0]) + (tmp['LBigToe'][0] - tmp['LHeel'][0]),
        #     (tmp['RBigToe'][1] - tmp['RHeel'][1]) + (tmp['LBigToe'][1] - tmp['LHeel'][1])
        # ]
        # o3 = np.arctan2(f0[0], f0[1]) + math.pi
        '''
        if f0[1] >= 0 and f0[0] < 0:
            o3 = np.arctan2(f0[1], f0[0]) + math.pi
        elif f0[1] < 0 and f0[0] < 0:
            o3 = np.arctan2(f0[1], f0[0]) - math.pi
        else:
            f0[1] >= 0 and f0[0] < 0
        '''
        df_orient01.loc[i] = o1
        df_orient02.loc[i] = o2
        df_orient03.loc[i] = o3
# 将计算出的不同部分朝向角度信息合并到一个 DataFrame df 中
    df['orient01'] = df_orient01['orient']
    df['orient02'] = df_orient02['orient']
    # df['orient03'] = df_orient03
    return df
