# Import modules
import cv2
import numpy as np
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default='./resoureces/video02.mp4')
parser.add_argument('--verbose', help="To print statements", default=True)
parser.add_argument('--coordinates',help='Provide 4 coordinate pairs (top-left, top-right, bottom-right, bottom-left) for perspective transformation')
args = parser.parse_args()

#Load pre-trained weights
def load_yolo():
    net = cv2.dnn.readNetFromDarknet("./models/yolov4.cfg", "./models/yolov4.weights")
    classes = []
    with open('./models/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def display_blob(blob):
    '''
    Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            # Restrict results to PERSON class only
            if conf > 0.3 and class_id==0:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img,loclist0,loclist1,initialized, player1, player2):
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.5)
        font = cv2.FONT_HERSHEY_PLAIN

        # Initialize player 1 and player 2 bounding boxes
        if initialized==0 and len(indexes)==2:
            x, y, w, h = boxes[indexes[0][0]]
            player1 = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            player1mean = player1.mean(axis=0).mean(axis=0)

            x, y, w, h = boxes[indexes[1][0]]
            player2 = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            player2mean = player2.mean(axis=0).mean(axis=0)

            initialized+=1

        if (len(indexes)==2):
            # Loop through both boxes
            x, y, w, h = boxes[indexes[0][0]]
            boxcolor = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            boxcolormean = boxcolor.mean(axis=0).mean(axis=0)

            diff1 = ssim(cv2.resize(player1,(200,200)),cv2.resize(boxcolor,(200,200)), multichannel=True)
            diff2 = ssim(cv2.resize(player2,(200,200)),cv2.resize(boxcolor,(200,200)), multichannel=True)

            if diff1>diff2:
                player1 = boxcolor
                player1mean=boxcolormean
                color = np.array([0.0,0.0,255.0])
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, 'Player 1', (x, y - 5), font, 1, color, 1)

                loc0 = (int(x+w/2), y + h+5)
                loclist0.append(loc0)
                clr = np.array([0.0,0.0,255.0])
                for l in range(len(loclist0)):
                    cv2.circle(img, loclist0[l], 3, clr,thickness=cv2.FILLED)

                x, y, w, h = boxes[indexes[1][0]]
                color = np.array([0.0,255.0,0.0])
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, 'Player 2', (x, y - 5), font, 1, color, 1)

                loc1 = (int(x+w/2), y + h+5)
                loclist1.append(loc1)
                clr = np.array([0.0,255.0,0.0])
                for l in range(len(loclist1)):
                    cv2.circle(img, loclist1[l], 3, clr,thickness=cv2.FILLED)
            else:
                player2 = boxcolor
                player2mean=boxcolormean
                color = np.array([0.0,255.0,0.0])
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, 'Player 2', (x, y - 5), font, 1, color, 1)

                loc1 = (int(x+w/2), y + h+5)
                loclist1.append(loc1)
                clr = np.array([0.0,255.0,0.0])
                for l in range(len(loclist1)):
                    cv2.circle(img, loclist1[l], 3, clr,thickness=cv2.FILLED)

                x, y, w, h = boxes[indexes[1][0]]
                color = np.array([0.0,0.0,255.0])
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, 'Player 1', (x, y - 5), font, 1, color, 1)

                loc0 = (int(x+w/2), y + h+5)
                loclist0.append(loc0)
                clr = np.array([0.0,0.0,255.0])
                for l in range(len(loclist0)):
                    cv2.circle(img, loclist0[l], 3, clr,thickness=cv2.FILLED)

        cv2.imshow("Image", img)
        t = time.time()
        #img_path = './player_tracking/'+str(t)+'.jpg'
        #cv2.imwrite(img_path,img)

        return initialized, player1, player2

def draw_fps(img,fps_time):
        font = cv2.FONT_HERSHEY_PLAIN
        color = np.array([255.0,255.0,255.0])
        cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 30),  font, 1, color, 2)
        fps_time = time.time()
        cv2.imshow("Image", img)

def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    loclist0=[]
    loclist1=[]
    ii=0
    initialized=0
    player1=np.zeros((200,200,3))
    player2=np.ones((200,200,3))
    while True:
                fps_time=time.time()
                _, frame = cap.read()

                ii+=1
                if frame is None:# or ii>=50:
                    break
                height, width, channels = frame.shape
                blob, outputs = detect_objects(frame, model, output_layers)
                boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
                initialized, player1, player2 = draw_labels(boxes, confs, colors, class_ids, classes, frame,loclist0,loclist1,initialized, player1, player2)
                draw_fps(frame,fps_time)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

    cap.release()
    return loclist0, loclist1

def plot_path(path0, path1, coordinates):
    #Create the plots
    path0_df = pd.DataFrame(path0, columns=['X','Y'])
    path1_df = pd.DataFrame(path1, columns=['X','Y'])

    # Floor coordinates from a frame
    c = coordinates.replace('[','').replace(']','').split(',')
    tl = [int(c[0]),int(c[1])]
    tr = [int(c[2]),int(c[3])]
    br = [int(c[4]),int(c[5])]
    bl = [int(c[6]),int(c[7])]

    pts1 = np.float32([tl, tr, br, bl])
    # Width and height values are based on regulation size squash court measurements in centimeters
    width, height = 640,544
    pts2 = np.float32([[0,0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Print the paths on generic court image
    court = cv2.imread('./resources/court.png')
    out0, out1 = [],[]
    for p0 in range(len(path0_df)):
        x,y = path0_df.iloc[p0,0], path0_df.iloc[p0,1]
        pts = np.array([[[x,y]]], dtype = "float32")
        warped = cv2.perspectiveTransform(pts,matrix)
        cv2.circle(court, (warped[0][0][0],warped[0][0][1]),5,(0,0,255),cv2.FILLED)
        out0.append([warped[0][0][0],warped[0][0][1]])
    for p1 in range(len(path1_df)):
        x,y = path1_df.iloc[p1,0], path1_df.iloc[p1,1]
        pts = np.array([[[x,y]]], dtype = "float32")
        warped = cv2.perspectiveTransform(pts,matrix)
        cv2.circle(court, (warped[0][0][0],warped[0][0][1]),5,(0,255,0),cv2.FILLED)
        out1.append([warped[0][0][0],warped[0][0][1]])

    court_with_path = './output/court_with_path.png'
    cv2.imwrite(court_with_path,court)
    print('Court with path stored at:',court_with_path,'Press ESCAPE to continue.')
    cv2.putText(court,'Player Paths',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA, False)
    cv2.imshow('court', court)
    cv2.waitKey(0)

    transformed_path0 = pd.DataFrame(out0, columns=['X','Y'])
    transformed_path1 = pd.DataFrame(out1, columns=['X','Y'])

    return transformed_path0, transformed_path1

def create_heatmap(transformed_path0, transformed_path1):

    transformed_paths = [transformed_path0, transformed_path1]
    i=1
    for path in transformed_paths:
        n = 30
        path = ((path/n).astype(int))*n
        path['Z']=100
        path = path.groupby(['X','Y'], as_index=False)['Z'].sum()
        colmax = max(path['Z'])
        colmin = min(path['Z'])

        court = cv2.imread('./resources/court.png')
        player = 'Player'+str(i) + 'heatmap'
        i+=1
        cv2.putText(court,player,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA, False)
        for p in range(len(path)):
            x = path.loc[p,'X']
            y = path.loc[p,'Y']
            z = (path.loc[p,'Z']-colmin)/(colmax-colmin)

            cv2.rectangle(court,(x,y),(x+n, y+n),(0,(1-z)*255,255),-1)


        outputfile = './output/'+ str(player) + '_path.png'
        cv2.imwrite(outputfile,court)
        print(outputfile,'written. Press ESCAPE to continue.')
        cv2.imshow('heatmap', court)
        cv2.waitKey(0)

if __name__ == '__main__':

    video_path = args.video_path
    coordinates= args.coordinates

    if args.verbose:
        print('Opening '+video_path+" .... Press ESCAPE to move to next steps.")

    # Start video, recognize players and plot their path
    path0, path1 = start_video(video_path)

    #Transform path & plot on 2D, save 2D image
    transformed_path0, transformed_path1 = plot_path(path0, path1, coordinates)

    # Create heat map
    create_heatmap(transformed_path0, transformed_path1)

    cv2.destroyAllWindows()
