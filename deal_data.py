import numpy as np
import pandas as pd
import os
from realize import *
from realize.upper_lower import *
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from realize.realize_data import Coordinate_Neck, limb_angle, body_orient
from realize.upper_lower import Shoulder_Hip_angle

x = np.array([0, 2, 3, 4, 7, 9, 2, 1, 2, 1]).reshape(-1, 1)
y = np.array([0, 1, 1, 1, 1, 2, 3, 3, 4, 7, 8, 9, 1, 1, 1, 1]).reshape(-1, 1)


class PersonReg():

    def __init__(self):
        # STEP 2: Create an ObjectDetector object.
        self.base_options = python.BaseOptions(
            model_asset_path="E:/FJNU/project/PoseDetection/score_v1/data/efficientdet_lite0.tflite")
        self.options = vision.ObjectDetectorOptions(base_options=self.base_options,
                                                    score_threshold=0.5)
        self.detector = vision.ObjectDetector.create_from_options(self.options)

    def visualize(
            self,
            image,
            detection_result
    ) -> np.ndarray:
        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (255, 0, 0)  # red
        """Draws bounding boxes on the input image and return it.
        Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
        Returns:
        Image with bounding boxes.
        """
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            # print(category_name)
            if (category_name == "person"):
                return True
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return False

    def isperson(self, image):

        # STEP 3: Load the input image.
        # image = mp.Image.create_from_file(image_path)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # STEP 4: Detect objects in the input image.
        detection_result = self.detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        image_copy = np.copy(image.numpy_view())
        annotated_image = self.visualize(image_copy, detection_result)

        # self.detector.close()
        return annotated_image


def get_lable(path):
    allpath = []
    dirs = os.listdir(path)
    for a in dirs:
        if os.path.isfile(path + '/' + a):
            allpath.append(dirs)
    return allpath


def deal_data_(data1,path):
    """
    特征提取
    """
    # data2 = pd.read_csv('./data/output2.csv')
    # 以提取与颈部相关的坐标或其他特征数据
    data1 = data1.reset_index(drop=True)
    neck_data = Coordinate_Neck(data1)
    # print(neck_data)
    # neck_data.to_csv('./data/test0.csv', index=None)

    # 深复制，浅复制原来的值会被改变
    # 计算重心
    # df1 = neck_data.copy(deep=True)
    # gravity_data = center_gravity(df1)
    # print(gravity_data)
    # print(neck_data)
    df2 = neck_data.copy(deep=True)
    limb_angle_data = limb_angle(df2)
    # print(limb_angle_data)
    # 计算身体朝向
    df3 = neck_data.copy(deep=True)
    body_orient_data = body_orient(df3)
    # print(body_orient_data)
    # df4 = neck_data.copy(deep=True)
    # arm_leg_data = arm_leg_angle(df4)
    # print(arm_leg_data)

    # 计算肩部和髋部之间的角度
    df5 = neck_data.copy(deep=True)
    Shoulder_Hip_data = Shoulder_Hip_angle(df5)
    # print(Shoulder_Hip_data)

    # df6 = neck_data.copy(deep=True)
    # Open_arm_data = open_arm(df6)
    # print(Open_arm_data)

    df_res = pd.concat(
        [limb_angle_data, body_orient_data, Shoulder_Hip_data], axis=1)

    # print(df_res)
    # df4.to_csv('E:/学习/python/python code/dtw_pose/data/f_'+path_name[11:]+'.csv', index=None)
    return df_res
