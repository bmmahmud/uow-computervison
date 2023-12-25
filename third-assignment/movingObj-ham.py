import math
import cv2
import numpy as np
import sys

"""
The Video is read in frames
The frame is then converted into a blob using the dnn.blobFromImage()
This is then later fed into the model and the output is detected
Since we need only the humans, a check for class 1 is done and a confidence of above 30% is taken into account
Later bounding boxes are drawn accordingly.
Eucledian distance is calculated for the objects based on its co-ordinates, which is then sorted based on its distance.
The first 3 objects out of them are taken into consideration.
"""


def downScaleImage(image,max_width, max_height):
    """
    Scales the image within a max width and height ceil
    Also maintains the aspect ratio
    """
    imgWidth = image.shape[1]
    imgHeight = image.shape[0]

    fx = max_width / imgWidth
    fy = max_height / imgHeight

    if(fx<fy):
        scalingFactor=fx
    else:
        scalingFactor=fy

    image = cv2.resize(image,(0,0),fx=scalingFactor,fy=scalingFactor,interpolation=cv2.INTER_LINEAR)
    return image

def task1(vdoPath):
    vid_forBG = cv2.VideoCapture(vdoPath)
    f_itr = 0

    #Background estimation
    frame_samples = vid_forBG.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    framesArr = []

    for frame_itr in frame_samples:
        vid_forBG.set(cv2.CAP_PROP_POS_FRAMES, frame_itr)
        _, frame = vid_forBG.read()
        frame = downScaleImage(frame,480,600)
        framesArr.append(frame)

    bGround = np.median(framesArr, axis=0).astype(dtype=np.uint8)

    #Creating GMM model
    gmm_model = cv2.createBackgroundSubtractorMOG2(varThreshold=25, detectShadows=False)

    capture = cv2.VideoCapture(vdoPath)
    
    while(True):
        _, frame = capture.read()
        frame = downScaleImage(frame,480,600)

        if (frame is None) :
            break

        #mask creation
        foreGrndMsk = gmm_model.apply(frame)

        #Noise removal
        kernel = np.ones((3, 3), np.uint8)
        img_mask = foreGrndMsk > 0
        filter = np.zeros_like(frame, np.uint8)
        filter[img_mask] = frame[img_mask]
        openFrame = cv2.morphologyEx(filter, cv2.MORPH_OPEN, kernel)
        closeFrame = cv2.morphologyEx(openFrame, cv2.MORPH_CLOSE, kernel)

        #stacking
        hstack1 = np.hstack([frame, bGround])
        hstack2 = np.hstack([np.repeat(foreGrndMsk[:, :, np.newaxis], 3, axis=2), closeFrame])

        window = np.vstack([hstack1, hstack2])

        cv2.imshow('OP', window)

        #Detection for connected components
        kernel_2 = np.ones((100, 100), np.uint8)
        frameGray = cv2.cvtColor(closeFrame, cv2.COLOR_BGR2GRAY)
        openFrameGray= cv2.dilate(frameGray, kernel_2, iterations=1)
        closeFrameGray = cv2.erode(openFrameGray, kernel_2, iterations=1)
        final_frame = cv2.medianBlur(closeFrameGray, 7)

        _, threshold = cv2.threshold(final_frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #CCA
        connections = 8
        op = cv2.connectedComponentsWithStats(threshold,connections,cv2.CV_32S)
        labels = op[0]
        stats = op[2]

        #print objects on console
        f_itr = f_itr+1
        car_object = 0
        person_object = 0
        other_object = 0
        for x in range(labels):
            if x != 0:
                width = stats[x][2]
                height = stats[x][3]
                if (width > height) and width > 40 and height > 30:
                    car_object = car_object+1
                elif (height > width) and height > 20:
                    person_object = person_object+1
                else:
                    other_object = other_object+1
        if (labels-1) > 0:
            print("Frame {0} : {1} objects ( {2} persons,{3} cars and {4} others )".format(f_itr,labels-1,person_object, car_object, other_object))
        else:
            print("Frame {0} : 0 objects".format(f_itr))

        #Enter to quit
        if cv2.waitKey(10) == 13:
            break

    capture.release()
    cv2.destroyAllWindows()

def task2(vdoPath):
    dnn = cv2.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework='TensorFlow')
    vdo = cv2.VideoCapture(vdoPath)

    while True:
        _, frame1 = vdo.read()
        _, frame2 = vdo.read()
        _, frame3 = vdo.read()
        _, frame4 = vdo.read()

        if(frame1 is None) or (frame2 is None) or (frame3 is None) or (frame4 is None):
            break

        frame1 = downScaleImage(frame1,600,400)
        frame2 = downScaleImage(frame2,600,400)
        frame3 = downScaleImage(frame3,600,400)
        frame4 = downScaleImage(frame4,600,400)

        if _:
            im1 = frame2
            im2 = frame3
            im3 = frame4

            #blob creation
            blob = cv2.dnn.blobFromImage(image=im1, size=(300, 300), mean=(104, 117, 123), swapRB=True)

            dnn.setInput(blob)
            op = dnn.forward()

            itr = 1 
            closeObjs = {}
            for detection in op[0, 0, :, :]:
               confidence = detection[2]
               class_id = detection[1]
                #Check for confidence threshold before drawing box
               if confidence > .35 and class_id == 1:
                   color = (0, 255, 225, 3)
                   #Calc x and y of the corner of the bounding box
                   x = int(detection[3] * im1.shape[1])
                   y = int(detection[4] * im1.shape[0])

                   #Calc the height and breath of the box
                   _width = int(detection[5] * im1.shape[1])
                   _height = int(detection[6] * im1.shape[0])

                   cv2.rectangle(im1,(x, y), (_width,_height), color, thickness=1)
                   cv2.rectangle(im2,(x, y), (_width, _height), color, thickness=1)

                   distance = int(math.dist((x,y),(600,400)))
                   closeObjs[distance]= (x,y,_width,_height)

                   #append class name
                   cv2.putText(im2, f'{itr}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,2,225), 2)
                   
                   itr+=1
            
            closeObjs_sorted = sorted(closeObjs.items())
            count = 1
            for obj in closeObjs_sorted:
                if count <= 3:
                    #display distances from camera to detected object
                    cv2.rectangle(im3, (int(obj[1][0]), int(obj[1][1])), (int(obj[1][2]), int(obj[1][3])), (0, 255, 1, 3),2)
                    cv2.putText(im3, 'distance = {}'.format(int(obj[0])), (int(obj[1][0]), int(obj[1][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(200,0,1), 2)
                else :
                    break
                count += 1
            
            window = np.vstack([np.hstack([frame1, im1]), np.hstack([im2, im3])])

            cv2.imshow("OP", window)
            #Enter to quit
            if cv2.waitKey(75) == 13:
                break
        else:
            break

    vdo.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if sys.argv[1] == "-b":
        task1(sys.argv[2])
    elif sys.argv[1] == "-d":
        task2(sys.argv[2])