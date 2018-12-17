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
	for x, y in normalized_contour[0]:
		if(minx > x):
			minx = x
		if(maxx < x):
			maxx = x 
	
	if(abs(minx) > abs(maxx)):
		return "TAB"
	else:
		return "HOLE"

"""
	This function takes in a vector of points, and transforms it so that it starts at the origin,
	and ends on the y-axis
"""
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
		ret_contour.append([(new_x, new_y)])
	
	return np.float32(np.asarray(ret_contour))

def create_edge(contour, start, end):
	#original
	contour_edge = contour[start:end]
	#Normalized contours are used for comparisons
	normalized_contour = normalize(contour_edge)
	reverse_contour = contour_edge[::-1]
	#same as normalized contour, but flipped 180 degrees
	reverse_normalized_contour = normalize(reverse_contour)
	edge_type = classify(normalized_contour)
	return (contour_edge, normalized_contour, reverse_normalized_contour, edge_type)

"""
This comparison iterates over every point in "this" contour,
finds the closest point in "that" contour and sums those distances up.
The end result is the sum divided by length of the 2 contours
"""
def compare2(edge1, edge2):
    #Return large number if an impossible situation is happening
    _,_,_,edge1_type = edge1
    _,_,_,edge2_type = edge2
    print edge1_type, edge2_type
    if(edge1_type == "OUTER_EDGE" or edge2_type == "OUTER_EDGE"):
    	return 100000000
    if(edge1_type == edge2_type):
    	return 100000000;
	"""    	
    cost=0
    double total_length =  cv::arcLength(normalized_contour, false) + cv::arcLength(that.reverse_normalized_contour, false);
    
    for(std::vector<cv::Point2f>::iterator i = normalized_contour.begin(); i!=normalized_contour.end(); i++){
        double min = 10000000;
        for(std::vector<cv::Point2f>::iterator j = that.reverse_normalized_contour.begin(); j!=that.reverse_normalized_contour.end(); j++){
            double dist = std::sqrt(std::pow(i->x - j->x,2) + std::pow(i->y - j->y, 2));
            if(dist<min) min = dist;
        }
        
        cost+=min;
    }
    return cost/total_length;
    """ 
    return 0
