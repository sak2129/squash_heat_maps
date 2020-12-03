# Import modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--video_path', help="Path of video file", default='./resources/video01.mp4')
args = parser.parse_args()

# Open video file and save the first frame in output folder
video_path = args.video_path
cap = cv2.VideoCapture(video_path)
_, frame = cap.read()

write_path = './output/videoframe.png'
cv2.imwrite(write_path,frame)
cap.release()
cv2.destroyAllWindows()
print('A picture frame has been stored in:',write_path,'.Please obtain vertices from that file and feed to heat map creation program.')
