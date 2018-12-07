import cv2
import os
import numpy as np

def color_to_bw(color, threshold):
	black_and_white = []
	for image in color:
		bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, bw = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)
		black_and_white.append(bw)
	return black_and_white

def median_blur(to_filter, k):
	ret = []
	for image in to_filter:
		median = cv2.medianBlur(image, k)
		ret.append(median)
	return ret

def getImages(path):
	images = []
	files = os.listdir(path)
	for filename in files:
		image = cv2.imread(os.path.join(path, filename))
		images.append(image)
	return images

def rotate(myList, pos): 
	return np.concatenate((myList[pos:], myList[:pos]), axis=0)

#This function takes in the beginning and ending of one vector, and returns
#an iterator representing the point where the first item in the second vector is.
def find_first_in(contour, corners):
	for i in range(len(contour)):	
		for j in range(len(corners)):
			if ( contour[i][0][0] == corners[j][0][0] and contour[i][0][1] == corners[j][0][1]):
				return i
	return len(contour)

#This returns iterators from the first vector where the value is equal places in the second vector.
def find_all_in(contour, corners):
	places = []
	for i in range(len(contour)):
		for j in range(len(corners)):
			if ( contour[i][0][0] == corners[j][0][0] and contour[i][0][1] == corners[j][0][1]):
				places.append(i)
	return places

def remove_duplicates(vec):
	vec = vec.tolist()
	dupes_found = True;
	while(dupes_found):
		dupes_found = False;
		dup_at = -1;
		for i in range(len(vec)):
			for j in range(len(vec)):                    
				if(j==i):
					continue          
				if(vec[i][0][0] == vec[j][0][0] and vec[i][0][1] == vec[j][0][1]):
					dup_at = j
					dupes_found = True;
					del vec[j]
					break
			if(dupes_found):
				break
	return  np.array(vec)

#Euclidian distance between 2 points.
def distance(a, b):
	return cv2.norm(a-b)

def translate_contour(cnt, offset_x, offset_y):
	ret_contour = []
	offset = (offset_x,offset_y)

	for i in range(len(cnt)):
		x = int(cnt[i][0][0]+offset_x+0.5)
		y = int(cnt[i][0][1]+offset_y+0.5)
		ret_contour.append((x,y))    
	return ret_contour;

def filter(to_filter, size):
	morphology = []
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
	for image in to_filter:
		image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
		image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
		morphology.append(image)
	return morphology

def draw_points(image, points, color = [0,0,255]):
	for p in points:
		cv2.circle(image,(p[0][0],p[0][1]),3,color,-1)
	return image;
