from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print(" #######loading model...#########################")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print(" #######Starting video stream...#################")
vs = VideoStream(src=0).start()
# warm up the camera 
time.sleep(2.0)
fps = FPS().start()
while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	print(frame.shape) # (225, 400, 3)
	(h, w) = frame.shape[:2]
	resized_image = cv2.resize(frame, (300, 300))
	blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
	net.setInput(blob)
	predictions = net.forward()
	# loop over the predictions
	for i in np.arange(0, predictions.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		# predictions.shape[2] = 100 here
		confidence = predictions[0, 0, i, 2]
		# Filter out predictions lesser than the minimum confidence level
		# Here, we set the default confidence as 0.2. Anything lesser than 0.2 will be filtered
		if confidence > args["confidence"]:
			# extract the index of the class label from the 'predictions'
			# idx is the index of the class label
			# E.g. for person, idx = 15, for chair, idx = 9, etc.
			idx = int(predictions[0, 0, i, 1])
			# then compute the (x, y)-coordinates of the bounding box for the object
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			# Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
			# Convert them to integers: 130 76 393 224
			(startX, startY, endX, endY) = box.astype("int")

			# Get the label with the confidence score
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("Object detected: ", label)
			# Draw a rectangle across the boundary of the objectq
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# Press 'q' key to break the loop
	if key == ord("q"):
		break

	fps.update()

fps.stop()

print(" ########Elapsed Time: {:.2f}#######".format(fps.elapsed()))
print(" ########Approximate FPS: {:.2f}####".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
