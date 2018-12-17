from __future__ import division
import cv2
import utils
import edge

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

def extract_edges(bw, corners):
	#Extract the contour,
	#TODO: probably should have this passed in from the puzzle, since it already does this
	#It was done this way b/c the contours don't correspond to the correct pixel locations
	#in this cropped version of the image.
	(_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	if  len(cnts) != 1:
		raise Exception('Found incorrect number of contours.')

	contour = cnts[0];

	contour = utils.remove_duplicates(contour);

	#out of all of the found corners, find the closest points in the contour,
	#these will become the endpoints of the edges    
	for i in range(len(corners)):
		best = 10000000000
		closest_point = contour[0]
		for j in range(len(contour)):        
			d = utils.distance(corners[i],contour[j])
			if d<best:
				best = d
				closest_point = contour[j]
		corners[i] = closest_point


	#We need the begining of the vector to correspond to the begining of an edge.
	contour = utils.rotate(contour, utils.find_first_in(contour, corners));
	
	#assert(corners[0]!=corners[1] && corners[0]!=corners[2] && corners[0]!=corners[3] && corners[1]!=corners[2] &&
	#       corners[1]!=corners[3] && corners[2]!=corners[3]);


	
	#std::vector<std::vector<cv::Point>::iterator> sections;
	sections = utils.find_all_in(contour, corners)


	#Make corners go in the correct order    
	for i in range(4):
		corners[i]*=sections[i];
	

	
	#assert(corners[1]!=corners[0] && corners[0]!=corners[2] && corners[0]!=corners[3] && corners[1]!=corners[2] &&
	#       corners[1]!=corners[3] && corners[2]!=corners[3]);
	
	edge1 = edge.create_edge(contour, sections[0], sections[1])
	edge2 = edge.create_edge(contour, sections[1], sections[2])
	edge3 = edge.create_edge(contour, sections[2], sections[3])
	edge4 = edge.create_edge(contour, sections[3], len(contour))
	return (edge1, edge2, edge3, edge4)

"""
	Classify the type of piece
"""
def classify(edges):
    count = 0
    for i in range(4):
        if(edges[i][3] == "OUTER_EDGE"):
        	count += 1
    
    if(count == 0):
        piece_type = "MIDDLE"
    elif (count == 1):
        piece_type = "FRAME"
    elif (count == 2):
        piece_type = "CORNER"
    else:
        raise Exception('Proble, found too many outer edges for this piece')
    return piece_type


def create_piece(color, black_and_white, estimated_piece_size):
	corners = find_corners(estimated_piece_size, black_and_white)
	edges = extract_edges(black_and_white, corners.copy())
	piece_type = classify(edges)
	#processed_image = utils.draw_points(color, edges[0],[0,255,255])
	#processed_image = utils.draw_points(processed_image, edges[1],[255,0,255])
	#processed_image = utils.draw_points(processed_image, edges[2],[255,0,0])
	#processed_image = utils.draw_points(processed_image, edges[3],[0,255,0])
	#processed_image = utils.draw_points(processed_image, corners,[0,0,255])	
	#cv2.imshow("Processed",processed_image)
	#cv2.waitKey(0)	
	return (edges, piece_type)