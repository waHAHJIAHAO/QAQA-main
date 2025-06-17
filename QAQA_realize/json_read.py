# 看那一列的差值会比较大
import pandas as pd
import copy
import json


def survey_data(x, y):
    Num_col = x.shape[1]
    lenx = x.shape[0] + 1
    leny = y.shape[0] + 1
    print('num_col', Num_col)
    print('lenx:', lenx)
    print('leny:', leny)
    lens = min(leny, lenx)
    a = [0.0 for x in range(0, Num_col + 1)]
    print(a)
    for i in range(lens - 1):
        # print(i)
        for j in range(Num_col):
            a[j] = a[j] + ((float(x.iloc[i][j]) - float(y.iloc[i][j])) ** 2)
    for i in range(Num_col):
        if i != 0 and i % 5 == 0:
            print()
        print(a[i], end='|')


def data_amplification(data):
    col = data.shape[0]

    df_1 = data['gravity_angle']

    for i in range(col):
        df_1[i] = df_1[i] * 100

    data['gravity_angle'] = df_1

    return data


# if __name__ == '__main__':
def json_read_(json_path):
    # json_path = "../data/08_gym1_1.json"
    # 假设您已经将 JSON 数据保存到一个名为 data.json 的文件中
    with open(json_path) as f:
        data = json.load(f)
    # 从 JSON 数据中提取 keypoints 并转换为 Pandas 数据框
    df = pd.json_normalize(data, record_path='keypoints')

    df = df.drop(columns='score')

    # 输出数据框
    # print(df)
    col_name = df['part'].unique()
    new_df = pd.DataFrame(columns=col_name)
    len_x = df.shape[0]
    for i in range(len_x):
        tmp = df.iloc[i]
        if tmp['part'] == "nose":
            new_data = {'nose': [1.1, 2.2],
                        'leftEye': [1.1, 2.2],
                        'rightEye': [1.1, 2.2],
                        'leftEar': [1.1, 2.2],
                        'rightEar': [1.1, 2.2],
                        'leftShoulder': [1.1, 2.2],
                        'rightShoulder': [1.1, 2.2],
                        'leftElbow': [1.1, 2.2],
                        'rightElbow': [1.1, 2.2],
                        'leftWrist': [1.1, 2.2],
                        'rightWrist': [1.1, 2.2],
                        'leftHip': [1.1, 2.2],
                        'rightHip': [1.1, 2.2],
                        'leftKnee': [1.1, 2.2],
                        'rightKnee': [1.1, 2.2],
                        'leftAnkle': [1.1, 2.2],
                        'rightAnkle': [1.1, 2.2]}
        new_data[tmp['part']][0] = tmp['position.x']
        new_data[tmp['part']][1] = tmp['position.y']

        if tmp['part'] == "rightAnkle":
            new_df.loc[len(new_df)] = new_data
    col_name = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder',
                'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
                'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    # 将原始列名和新列名转换为字典
    new_col_name = {new_df.columns[i]: col_name[i] for i in range(len(new_df.columns))}

    # 更改 Pandas 数据框的列名
    new_df = new_df.rename(columns=new_col_name)
    # new_df.to_csv('./test_output.csv')
    return new_df


def data_read_(data):
    """
     删除一些列，并且重新命名
    未修改之前
    df.columns = ['Nose', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
                  'LWrist', 'RWrist', 'LPinky', 'RPinky', 'LIndex',
                  'RIndex', 'LThumb', 'RThumb', 'LHip', 'RHip',
                  'LKnee', 'RKnee', 'LAnkle', 'RAnkle', 'LHeel',
                  'RHeel', 'LBigToe', 'RBigToe']
    mediapipe没有spine，thorax neck head这四个点
    mmpose 中把thorax换成neck，neck base换成nose
    丢掉head和spine ，root就是mid hip ，foot 记为ankle,最后mmpose提取的只剩15个keypoint
    """
    df = data
    df.drop(columns=['spine', 'neck_base'], inplace=True)
    df.columns = ['MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee',
                  'LAnkle', 'Neck', 'Nose', 'LShoulder', 'LElbow', 'LWrist',
                  'RShoulder', 'RElbow', 'RWrist']
    return df
