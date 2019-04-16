import argparse
import cv2

# create the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(parser.parse_args())

# load the input image
img = cv2.imread(args["image"])

# initialize OpenCV's static fine grained saliency detector and compute the saliency
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(img)

# show the images
cv2.imshow("Input", img)
cv2.imshow("Output", saliencyMap)
cv2.waitKey(0)