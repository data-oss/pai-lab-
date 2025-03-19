#python -m venv venv
#venv/Scripts/activate
#----------------------------------------
#|Set-ExecutionPolicy Unrestricted -Force|
#----------------------------------------
import cv2
import pandas 
import matplotlib.pyplot as plt
import numpy as np
import imutils 
from PIL import Image

## read image 

img = cv2.imread("dp pic.jpeg", cv2.IMREAD_COLOR) 
# show image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#----------------------------------------
# saving emige
cv2.imwrite("dp pic1.jpeg", img)  
#----------------------------------------
##add two images

img1 = cv2.imread("dp pic.jpeg", cv2.IMREAD_COLOR) 
img2 = cv2.imread("dp pic1.jpeg", cv2.IMREAD_COLOR) 
img3 = cv2.add(img1, img2)
cv2.imshow("image", img3)
cv2.waitKey(0)

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  
#----------------------------------------
# subtruct images

img4 = cv2.subtract(img1, img2)
cv2.imshow("image", img4)
cv2.waitKey(0)
#----------------------------------------
##Bitwise AND operation on Image:

img5 = cv2.bitwise_and(img1, img2)
cv2.imshow("image", img5)
cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  

#----------------------------------------
##Bitwise OR operation on Image:

img6 = cv2.bitwise_or(img1, img2)
cv2.imshow("image", img6)
cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
#----------------------------------------
# # Bitwise XOR operation on Image:

img7 = cv2.bitwise_xor(img1, img2)
cv2.imshow("image", img7)
cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
#----------------------------------------
## Bitwise NOT operation on Image:

img8 = cv2.bitwise_not(img1)
cv2.imshow("image", img8)
cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
#----------------------------------------

###  ------Image Processing ----
# #image resizing
half = cv2.resize(img1, (0, 0), fx = 0.1, fy = 0.1)
bigger = cv2.resize(img1, (1050, 1610))

stretch_near = cv2.resize(img1, (780, 540), 
               interpolation = cv2.INTER_LINEAR)

Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
images =[img1, half, bigger, stretch_near]
count = 4
for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])

plt.show()

#----------------------------------------
## Eroding an image

image = cv2.imread("E:\PAI\pai lab\opncv\dp pic.jpeg")  
window_name = 'Image' 
kernel = np.ones((6, 5), np.uint8) 
image = cv2.erode(image, kernel)  
cv2.imshow(window_name, image)  
cv2.waitKey(0)

#----------------------------------------

## -----Image blurring ----
image = cv2.imread('E:\PAI\pai lab\opncv\dp pic.jpeg') 
  
cv2.imshow('Original Image', image) 
cv2.waitKey(0) 
  
# Gaussian Blur 
Gaussian = cv2.GaussianBlur(image, (7, 7), 0) 
cv2.imshow('Gaussian Blurring', Gaussian) 
cv2.waitKey(0) 
  
# Median Blur 
median = cv2.medianBlur(image, 5) 
cv2.imshow('Median Blurring', median) 
cv2.waitKey(0) 
  
  
# Bilateral Blur 
bilateral = cv2.bilateralFilter(image, 9, 75, 75) 
cv2.imshow('Bilateral Blurring', bilateral) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

#----------------------------------------

##  analyze an image using Histogram
img = cv2.imread('E:\PAI\pai lab\opncv\dp pic.jpeg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Histogram Calculation
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Histogram Display
plt.plot(hist)
plt.show()

#----------------------------------------
##  -----Image Thresholding ----
img = cv2.imread('E:\PAI\pai lab\opncv\dp pic.jpeg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Thresholding
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey(0)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Binary Inverse', thresh2)
cv2.waitKey(0)

#-------------------------------------------------------------
# ## cv2.cvtColor() method
img = cv2.imread('E:\PAI\pai lab\opncv\dp pic.jpeg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)

# Converting to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-------------------------------------------------------------
# ## cv2.split() method
img = cv2.imread('E:\PAI\pai lab\opncv\dp pic.jpeg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Splitting the image
b, g, r = cv2.split(img)

# Displaying the channels
cv2.imshow('Blue Channel', b)
cv2.waitKey(0)

cv2.imshow('Green Channel', g)
cv2.waitKey(0)

cv2.imshow('Red Channel', r)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-------------------------------------------------------------

# ## Bilateral Filtering
img = cv2.imread('E:\PAI\pai lab\opncv\dp pic.jpeg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Applying bilateral filter
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('Bilateral Filtered Image', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-------------------------------------------------------------
## ------Feature Detection and Description---
## ---- Circle Detection--

img = cv2.imread('photo.jpeg', cv2.IMREAD_COLOR) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray_blurred = cv2.blur(gray, (3, 3)) 
detected_circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 1, maxRadius = 40) 
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
    cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
    cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
    cv2.imshow("Detected Circle", img) 
    cv2.waitKey(0)

#-------------------------------------------------------------
## ----Detect corner of an image

img = cv2.imread('photo.jpeg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-------------------------------------------------------------
## ---Corner detection with Harris Corner Detection method 

image = cv2.imread('photo.jpeg')
operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
operatedImage = np.float32(operatedImage) 
dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
dest = cv2.dilate(dest, None)
image[dest > 0.01 * dest.max()]=[0, 0, 255]
cv2.imshow('Image with Borders', image)
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows()

#-------------------------------------------------------------
##---Find Circles and Ellipses in an Image

image = cv2.imread('E:\PAI\pai lab\opncv\photo.jpeg', 0)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True 
params.minCircularity = 0.9
params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image) 
blank = np.zeros((1, 1))  
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
number_of_blobs = len(keypoints) 
text = "Number of Circular Blobs: " + str(len(keypoints)) 
cv2.putText(blobs, text, (20, 550), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
cv2.imshow("Filtering Circular Blobs Only", blobs) 
cv2.waitKey(0) 
cv2.destroyAllWindows()


#-------------------------------------------------------------

##---Smile detection using

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml') 

def detect(gray, frame): 
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
	for (x, y, w, h) in faces: 
		cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
		roi_gray = gray[y:y + h, x:x + w] 
		roi_color = frame[y:y + h, x:x + w] 
		smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 

		for (sx, sy, sw, sh) in smiles: 
			cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
	return frame 

video_capture = cv2.VideoCapture(0) 
while video_capture.isOpened(): 

	_, frame = video_capture.read() 
				 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	 
	canvas = detect(gray, frame) 
					 
	cv2.imshow('Video', canvas) 

	if cv2.waitKey(1) & 0xff == ord('q'):			 
		break

video_capture.release()

cv2.destroyAllWindows() 

#-----------------------------------------------------------------

##--------     Drawing Functions
#   ------ rectangle
image = cv2.imread("photo.jpeg")
window_name = 'Image'
start_point = (0, 0)
end_point = (250, 250)
color = (255, 0, 0)
thickness = 5
image = cv2.rectangle(image, start_point, end_point, color, thickness)
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

###-----------------------------------------------------
#  ---arrowedLine
image = cv2.imread("photo.jpeg") 
window_name = "Image"
start_point = (0, 0)
end_point = (200, 200)
color = (0, 255, 0)  
thickness = 9
image = cv2.arrowedLine(image, start_point, end_point,color, thickness)  
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

###-----------------------------------------------------
##    ---ellipse

image = cv2.imread("photo.jpeg") 
window_name = "Image"
center_coordinates = (120, 100) 
axesLength = (100, 50) 
angle = 0
startAngle = 0
endAngle = 360
color = (0, 0, 255) 
thickness = 5
image = cv2.ellipse(image, center_coordinates, axesLength,angle, startAngle, endAngle, color, thickness)
cv2.imshow(window_name, image)  
cv2.waitKey(0)
cv2.destroyAllWindows()

##-----------------------------------------------------------------
#  ---circle

image = cv2.imread("photo.jpeg") 
window_name = "Image"
center_coordinates = (120, 100)
radius = 50
color = (0, 0, 255)
thickness = 5
image = cv2.circle(image, center_coordinates, radius, color, thickness)
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

##-----------------------------------------------------------------
# ---text string 

image = cv2.imread("photo.jpeg") 
window_name = "Image"
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (0, 0, 255)
thickness = 2
image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness)
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

##-----------------------------------------------------------------

#### Working with Videos--
# ----- Play a video

cap = cv2.VideoCapture('video.mp4')
if (cap.isOpened()== False):
    print("Error opening video file")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

#--------------------------------------

# --- Create video using multiple images 

images = [cv2.imread("photo.jpeg"), cv2.imread("photo.jpeg"), cv2.imread("photo.jpeg")]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
for image in images:
    out.write(image)
out.release()
cv2.destroyAllWindows()

#-------------------------------------- 
