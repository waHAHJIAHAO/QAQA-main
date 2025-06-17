from realize.realize import realize_dtw

from realize.json_read import *
from realize.keypoint import obtain_keyponint
from deal_data import *


def test_main():
    path_test = "./data/yjj_13.json"
    path_temp = "./data/yjj_23.json"
    # 这是用户上传的测试视频生成的json路径：
    datax = json_read_(path_test)
    # 这是模板视频生成的json路径：
    datay = json_read_(path_temp)
    datax = deal_data_(datax, path_test)
    datay = deal_data_(datay, path_temp)

    realize_dtw(datax, datay)
    # survey_data(datax, datay)


def test_keypoint():
    action_path = '../test.mp4'
    suffix = action_path[-3:]
    df = obtain_keyponint(action_path)
    print(df)


if __name__ == '__main__':
    test_keypoint()
