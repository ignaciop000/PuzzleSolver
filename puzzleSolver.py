from __future__ import division
import os
import cv2
import numpy as np
import math

def classify(normalized_contour):
	#See if it is an outer edge comparing the distance between beginning and end w/
	#the arc length.
	normalized_contour = np.array(normalized_contour)
	contour_length = cv2.arcLength(normalized_contour, False)
	begin_end_distance = cv2.norm(normalized_contour[0]-normalized_contour[-1])
	if(contour_length < begin_end_distance*1.3):        
		return "OUTER_EDGE"
	
	#Find the minimum or maximum value for x in the normalized contour and base
	#the classification on that
	minx  = 100000000
	maxx = -100000000
	for x, y in normalized_contour:
		if(minx > x):
			minx = x
		if(maxx < x):
			maxx = x 
	
	if(abs(minx) > abs(maxx)):
		return "TAB"
	else:
		return "HOLE"


def normalize(cont):
	ret_contour = []
	a = cont[0][0]
	b = cont[-1][0]

	#Calculating angle from vertical
	b = (b[0] - a[0],b[1] - a[1])
	
	theta = math.acos(b[1]/(cv2.norm(b)));
	if(b[0] < 0):
		theta = -theta
	
	#Theta is the angle every point needs rotated.
	#and -a is the translation
	for p in cont:
		p = p[0]
		#Apply translation
		temp_point = (p[0]-a[0],p[1]-a[1])
		#Apply roatation
		new_x = math.cos(theta) * temp_point[0] - math.sin(theta) * temp_point[1]
		new_y = math.sin(theta) * temp_point[0] + math.cos(theta) * temp_point[1]
		ret_contour.append((new_x, new_y))
	
	return ret_contour

def create_edge(contour, start, end):
	#original
	contour_edge = contour[start:end]
	#Normalized contours are used for comparisons
	normalized_contour = normalize(contour_edge)
	reverse_contour = contour_edge[::-1]
	#same as normalized contour, but flipped 180 degrees
	reverse_normalized_contour = normalize(reverse_contour)
	type = classify(normalized_contour)

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


def extract_edges(bw, corners):
	#Extract the contour,
	#TODO: probably should have this passed in from the puzzle, since it already does this
	#It was done this way b/c the contours don't correspond to the correct pixel locations
	#in this cropped version of the image.
	(_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	if  len(cnts) != 1:
		raise Exception('Found incorrect number of contours.')

	contour = cnts[0];

	contour = remove_duplicates(contour);

	#out of all of the found corners, find the closest points in the contour,
	#these will become the endpoints of the edges    
	for i in range(len(corners)):
		best = 10000000000
		closest_point = contour[0]
		for j in range(len(contour)):        
			d = distance(corners[i],contour[j])
			if d<best:
				best = d
				closest_point = contour[j]
		corners[i] = closest_point


	#We need the begining of the vector to correspond to the begining of an edge.
	contour = rotate(contour, find_first_in(contour, corners));
	
	#assert(corners[0]!=corners[1] && corners[0]!=corners[2] && corners[0]!=corners[3] && corners[1]!=corners[2] &&
	#       corners[1]!=corners[3] && corners[2]!=corners[3]);


	
	#std::vector<std::vector<cv::Point>::iterator> sections;
	sections = find_all_in(contour, corners)


	#Make corners go in the correct order    
	for i in range(4):
		corners[i]*=sections[i];
	

	
	#assert(corners[1]!=corners[0] && corners[0]!=corners[2] && corners[0]!=corners[3] && corners[1]!=corners[2] &&
	#       corners[1]!=corners[3] && corners[2]!=corners[3]);
	
	edge1 = create_edge(contour, sections[0], sections[1])
	edge2 = create_edge(contour, sections[1], sections[2])
	edge3 = create_edge(contour, sections[2], sections[3])
	edge4 = create_edge(contour, sections[3], len(contour))
	return (edge1, edge2, edge3, edge4)


def draw_points(image, points, color = [0,0,255]):
	for p in points:
		cv2.circle(image,(p[0][0],p[0][1]),3,color,-1)
	return image;

def find_corners(piece_size, black_and_white):
	minDistance = piece_size
	blockSize = 25
	useHarrisDetector = True
	k = 0.04
	minimun = 0
	maximun = 1
	max_iterations = 100
	found_all_corners = False
	corners = []
	while (0<max_iterations):
		max_iterations -= 1
		qualityLevel = float((minimun+maximun)/2)
		corners = cv2.goodFeaturesToTrack(black_and_white.copy(),100,qualityLevel,minDistance,None,blockSize,useHarrisDetector,k);
		if(len(corners) > 4):
			#Found too many corners increase quality
			minimun = qualityLevel;
		elif (len(corners) < 4):
			maximun = qualityLevel;
		else:
			#found all corners
			found_all_corners = True;
			break          


	#Find the sub-pixel locations of the corners.
	winSize = ( blockSize, blockSize )
	zeroZone = ( -1, -1 )
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
	#cv::TermCriteria criteria = cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
	
	#Calculate the refined corner locations
	cv2.cornerSubPix(black_and_white, corners, winSize, zeroZone, criteria)
		
	if not found_all_corners:
		raise Exception('Failed to find correct number of corners '+len(corners))

	return corners


def create_piece(color, black_and_white, estimated_piece_size):
	corners = find_corners(estimated_piece_size, black_and_white);
	#corners_image = draw_points(color, corners)
	#cv2.imshow("corners_image",corners_image)
	#cv2.waitKey(0)	
	edges = extract_edges(black_and_white, corners);
	print edges
	#classify();
	return (color, black_and_white, estimated_piece_size)

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

def extract_pieces(path, needs_filter, threshold, piece_size):
	pieces = []
	color_images = getImages(path)
	if needs_filter:
		blured_images = median_blur(color_images, 5)
		bws = color_to_bw(blured_images, threshold)
	else:
		bws = color_to_bw(color_images, threshold)
		bws = filter(bw,2)
	
	for bw, color_image in zip(bws,color_images):
		(_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		for contour in cnts:
			bordersize = 15
			x,y,w,h = cv2.boundingRect(contour)
			if (w < piece_size or h < piece_size):
				continue
			new_bw = np.zeros((h+2*bordersize,w+2*bordersize), dtype='uint8')
			contours_to_draw = []
			contours_to_draw.append(translate_contour(contour, bordersize-x, bordersize-y))			
			cv2.drawContours(new_bw,  np.asarray(contours_to_draw), -1, 255, -1)
			w += bordersize * 2
			h += bordersize * 2
			x -= bordersize
			y -= bordersize
			mini_color = color_image[y:y+h,x:x+w]
			mini_bw = new_bw
			mini_color = mini_color.copy()
			mini_bw = mini_bw.copy()

			piece = create_piece(mini_color, mini_bw, piece_size)
			pieces.append(piece)	
	return pieces

def puzzle(folderpath, estimated_piece_size, thresh, filter = True):
	pieces = extract_pieces(folderpath, filter, thresh, estimated_piece_size)	
	return pieces


if __name__ == "__main__":	
	pieces = puzzle("Scans/horses numbered", 380,  50)
