# import os
from ultralytics import YOLO
import cv2
import numpy as np
import time

# cap = cv2.VideoCapture('beeHive.mp4')
# ret, frame = cap.read()

# H, W, _ = frame.shape

# out = cv2.VideoWriter('beeHive_out.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# model = YOLO('beeModel2.pt')

# threshold = 0.5

# class_name_dict = {0: 'bee'}

# while ret:
#     results = model(frame)[0]

# #############################################################

# model = YOLO(r'C:\Users\Cavid\Desktop\AIProjects\best(1).pt') # 'beeModel2.pt'
model = YOLO('bbest.pt')

results = model(source=0, show=True, conf=0.3, save=True)




