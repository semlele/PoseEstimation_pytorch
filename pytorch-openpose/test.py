import os
import cv2
import copy
from src import util
from src.body import Body

INPUT_FILE_NAME = "demo.jpg"

if __name__ == "__main__":
    body_estimation = Body('model/body_pose_model.pth')

    target_image_path = 'images/' + INPUT_FILE_NAME
    oriImg = cv2.imread(target_image_path)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    basename_name = os.path.splitext(os.path.basename(target_image_path))[0]

    result_image_path = "result/pose_" + basename_name + ".jpg"
    cv2.imwrite(result_image_path, canvas)