import os
import cv2
import numpy as np
import math
import utils
import piece

def extract_pieces(path, needs_filter, threshold, piece_size):
	pieces = []
	color_images = utils.getImages(path)
	if needs_filter:
		blured_images = utils.median_blur(color_images, 5)
		bws = utils.color_to_bw(blured_images, threshold)
	else:
		bws = utils.color_to_bw(color_images, threshold)
		bws = utils.filter(bw,2)
	
	for bw, color_image in zip(bws,color_images):
		(_, cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		for contour in cnts:
			bordersize = 15
			x,y,w,h = cv2.boundingRect(contour)
			if (w < piece_size or h < piece_size):
				continue
			new_bw = np.zeros((h+2*bordersize,w+2*bordersize), dtype='uint8')
			contours_to_draw = []
			contours_to_draw.append(utils.translate_contour(contour, bordersize-x, bordersize-y))			
			cv2.drawContours(new_bw,  np.asarray(contours_to_draw), -1, 255, -1)
			w += bordersize * 2
			h += bordersize * 2
			x -= bordersize
			y -= bordersize
			mini_color = color_image[y:y+h,x:x+w]
			mini_bw = new_bw
			mini_color = mini_color.copy()
			mini_bw = mini_bw.copy()

			one_piece = piece.create_piece(mini_color, mini_bw, piece_size)
			pieces.append(one_piece)	
	return pieces

def puzzle(folderpath, estimated_piece_size, thresh, filter = True):
	pieces = extract_pieces(folderpath, filter, thresh, estimated_piece_size)	
	return pieces