import numpy as np
import cv2
import time
import sys
import copy
import constant as con

# pre defination
# cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def find_eye(frame, gray, faces):
	eye_region_width = faces[2] * (con.kEyePercentWidth/100.0)
	eye_region_height = faces[3] * (con.kEyePercentHeight/100.0)
	eye_region_top = faces[3] * (con.kEyePercentTop/100.0)
	leftEyeRegion = [int(faces[0]+faces[2]*(con.kEyePercentSide/100.0)), 
	                 int(faces[1]+eye_region_top), 
	                 int(eye_region_width), 
	                 int(eye_region_height)]
	rightEyeRegion = [int(faces[0]+faces[2] - eye_region_width - faces[2]*(con.kEyePercentSide/100.0)), 
	                  int(faces[1]+eye_region_top), 
	                  int(eye_region_width), 
	                  int(eye_region_height)]
	cv2.rectangle(frame,(leftEyeRegion[0], leftEyeRegion[1]),
                        (leftEyeRegion[0]+leftEyeRegion[2], leftEyeRegion[1]+leftEyeRegion[3]),
                        (0,255,0),2)
	cv2.rectangle(frame,(rightEyeRegion[0], rightEyeRegion[1]),
                        (rightEyeRegion[0] + rightEyeRegion[2], rightEyeRegion[1] + rightEyeRegion[3]),
                        (0,255,0),2)

	left_p = find_eye_center(gray, leftEyeRegion)
	right_p = find_eye_center(gray, rightEyeRegion)
	left_p[0] += leftEyeRegion[0]
	left_p[1] += leftEyeRegion[1]
	l = (int(left_p[0]), int(left_p[1]))
	right_p[0] += rightEyeRegion[0]
	right_p[1] += rightEyeRegion[1]
	r = (int(right_p[0]), int(right_p[1]))
	cv2.circle(frame, l, 5, (0,0,255), -1)
	cv2.circle(frame, r, 5, (0,0,255), -1)
	return frame

def unscalePoint(p, size):
	ratio = (float(con.kFastEyeWidth))/size[2]
	x = round(p[0] / ratio, 0)
	y = round(p[1] / ratio, 0)
	return (x,y)

def scaleToFastSize(frame):   
	return cv2.resize(frame, (con.kFastEyeWidth, int(con.kFastEyeWidth)))

def find_eye_center(gray, region):
	eye_area = gray[region[1]:region[1]+region[3], region[0]:region[0]+region[2]]
	eye_area = scaleToFastSize(eye_area)
	grdient_X = compute_gradient(eye_area)
	grdient_Y = cv2.transpose(compute_gradient(cv2.transpose(eye_area)))
	mags = matrixMagnitude(grdient_X, grdient_Y)
	gradientThresh = computeDynamicThreshold(mags, con.kGradientThreshold)
	(h, w) = eye_area.shape
	for y in range(h):
		for x in range(w):
			if mags[y][x] > gradientThresh:
				grdient_X[y][x] /= mags[y][x]
				grdient_Y[y][x] /= mags[y][x]
			else:
				grdient_X[y][x] = 0
				grdient_Y[y][x] = 0
	weight = cv2.GaussianBlur(eye_area,(con.kWeightBlurSize, con.kWeightBlurSize),0)
	(h, w) = weight.shape
	for y in range(h):
		for x in range(w):
			 weight[y][x] = 255 - weight[y][x]
	out_sum = np.zeros(eye_area.shape)
	for y in range(h):
		for x in range(w):
			if grdient_X[y][x] == 0 and grdient_Y[y][x] == 0:
				continue
			out_sum = testPossibleCentersFormulac(x, y, weight, grdient_X[y][x], grdient_Y[y][x], out_sum)
	out = out_sum / (h*w)
	[minVal, maxVal, minLoc, maxLoc] = cv2.minMaxLoc(out)
	maxLoc = unscalePoint(maxLoc, region)
	m = [maxLoc[0], maxLoc[1]]
	return m

def testPossibleCentersFormulac(x, y, weight, gX, gY, out):
	(h, w) = weight.shape
	for cy in range(h):
		for cx in range(w):
			if x == cx and y == cy:
				continue
			dx = x - cx
			dy = y - cy
			mag = np.sqrt(dx*dx + dy*dy)
			dx /= mag
			dy /= mag
			dotproduct = dx*gX + dy*gY
			dotproduct = max(0.0, dotproduct)
			if con.kEnableWeight: 
				out[cy][cx] += dotproduct * dotproduct * (weight[cy][cx]/con.kWeightDivisor)
			else:
				out[cy][cx] += dotproduct * dotproduct
	return out

def compute_gradient(frame):
	out = np.zeros(frame.shape)
	(h, w) = frame.shape[:2]
	for y in range(h):
		out[y][0] = int(frame[y][1]) - int(frame[y][0])
		for x in range(w-1):
			out[y][x] = (int(frame[y][x+1]) - int(frame[y][x-1]))/2.0
		out[y][w-1] = int(frame[y][w-1]) - int(frame[y][w-2])
	return out

def matrixMagnitude(grdient_X, grdient_Y):
	out = np.zeros(grdient_X.shape)
	(h, w) = grdient_X.shape
	for y in range(h):
		for x in range(w):
			out[y][x] = np.sqrt((grdient_X[y][x]*grdient_X[y][x])+(grdient_Y[y][x]*grdient_Y[y][x]))
	return out

def computeDynamicThreshold(mags, kGradientThreshold):     # didn't figure out 
	([[meanMagnGrad]], [[stdMagnGrad]]) = cv2.meanStdDev(mags)
	(h, w) = mags.shape
	stdDev = stdMagnGrad / np.sqrt(h*w)
	return kGradientThreshold * stdDev + meanMagnGrad

def main():
	# Capture frame-by-frame
	# while True:
	# ret, frame = cap.read()
	frame = cv2.imread("einstein.jpg", -1)

	# convert the frame to gray scale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect face with Cascade Classifier
	faces = face_cascade.detectMultiScale(gray, 
		                                  scaleFactor=1.1, 
	                                      minNeighbors=5)
	num_face = 0

	for (fx,fy,w,h) in faces:
		cv2.rectangle(frame,(fx,fy),(fx+w,fy+h),(255,0,0),2)
		# find eyes according to percentages 
		if w * h:
			frame = find_eye(frame, gray, faces[num_face])
		num_face += 1

	# Display the resulting frame
	cv2.imwrite("result.jpg", frame)
	# cv2.imshow('gray', frame)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break
	# When everything done, release the capture
	# cap.release()
	cv2.destroyAllWindows()


# if python says run, then we should run
if __name__ == '__main__':
	main()