import cv2
import numpy as np
from realsense_camera import *
import time
from imutils.video import FPS
import argparse
import glob
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-u", "--use-gpu", type=bool, default=0,
	help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of images")
args = vars(ap.parse_args())

# Generate random colors(80 colors equals to detectable classes of the model, 3 is the number of channels)
colors = np.random.randint(0, 255, (80, 3))

# Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow(
    "dnn/frozen_inference_graph_coco.pb", "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


# check if we are going to use GPU
if args["use_gpu"]:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing video stream...")

# Load Realsense camera
rs = RealsenseCamera()

fps = FPS().start()

images = {}

def hog():
    # loop over the images paths
    hist = []
    index = {}
    for imagePath in glob.glob(args["dataset"] + "\*.jpg"):
        # extract the image filename (assumed to be unique)
        # load the image, updating the images dictionary
        fileName = imagePath[imagePath.rfind("\\") + 1:]
        image = cv2.imread(imagePath)
        images[fileName] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # extract a 3D RFB color histogram from image,
        # using 8 bins per channel, normalize, and update the index

        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
        fileName = int(os.path.split(fileName)[1].split('.')[0])
        index[fileName] = hist
    return hist, index
        # print(hist.shape)


while True:
    # Load image
    # img = cv2.imread('road.jpg')
    # height, width, _ = img.shape
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    if not ret:
        rs.release()
        cv2.destroyAllWindows()
        break

    height, width, _ = bgr_frame.shape

    hog_results = {}

    hist, index = hog()

    # Create a background image
    background = np.zeros((height, width, 3), np.uint8)
    background[:] = (100, 100, 0)
    # Detect objects
    blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
    net.setInput(blob)
    start = time.time()
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()
    detection_count = boxes.shape[2]
    print("[Info] Mask R-CNN took {:.6f} seconds".format(end - start))

    for i in range(detection_count):
        # box = boxes[0, class, object detected]
        # Inside the array of the object detected:
        # index 1 is the class of the object
        # index 2 is the score
        # index 3 to 6 is the box coordinates
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]

        # If class is not person continue
        if class_id != 0:
            continue

        if score > 0.5:           

        
            # Get box coordinates
            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            roi = bgr_frame[y: y2, x: x2]

            roi_height, roi_width, _ = roi.shape
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), (255, 0, 0), 2)

            # Get the mask
            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            mask = (mask > 0.5)
            
            mask = np.stack((mask,) * 3, axis = -1)
            mask = mask.astype('uint8')
            bg = 255 - mask * 255
            mask_show = np.invert(bg)
            mask_img = roi * mask
            result = mask_img + bg

            new_ped_descriptor = cv2.calcHist([result], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            new_ped_descriptor = cv2.normalize(new_ped_descriptor, new_ped_descriptor).flatten()

            if hist != []:
                for (k, hist) in index.items():
                    d = cv2.compareHist(new_ped_descriptor, hist, cv2.HISTCMP_BHATTACHARYYA)
                    # There is a bug here that needs to be fixes
                    hog_results[k] = d

                hog_results = sorted([(v, k) for (k, v) in hog_results.items()])
                min_index = np.argmin(hog_results)
                min_distance = hog_results[min_index]

                if min_distance[0] <= 0.5:
                    print("Old pedestrian detected")
                    pred = "Prev. Pers. Det"
                else:
                    print("New pedestrian detected")
                    ID = len(index) + 1
                    print(ID)
                    hist = np.concatenate(
                            (hist, new_ped_descriptor), axis=0)
                    fileName = "dataset/" + str(ID) + ".jpg"
                    cv2.imwrite(fileName, result)    
                    pred = "New. Pers. Det"                
            else:
                print("New pedestrian detected")
                ID = len(index) +  1
                print(ID)
                hist = new_ped_descriptor
                fileName = "dataset/" + str(ID) + ".jpg"
                cv2.imwrite(fileName, result)
                pred = "New. Pers. Det"
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(bgr_frame, pred, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # print(box)

    cv2.imshow("Image", bgr_frame)
    # cv2.imshow("Image", final)
    key = cv2.waitKey(1)
    if key == 27:
        break

    # update the FPS counter
    fps.update()
    
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# rs.release()
cv2.destroyAllWindows()
