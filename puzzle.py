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

def fill_costs(pieces):
	no_edges = len(pieces)*4
	matches = []
	for i in range(no_edges):
		for j in range (i, no_edges):
			matches.append((i, j,))
"""
	
		for(int j=i; j<no_edges; j++){
			match_score score;
			score.edge1 =(int) i;
			score.edge2 =(int) j;
			score.score = pieces[i/4].edges[i%4].compare2(pieces[j/4].edges[j%4]);
{
			matches.push_back(score);
}
		}
	}
	std::sort(matches.begin(),matches.end(),match_score::compare);
"""


"""
	Solves the puzzle
"""
def solve(pieces):
	print "Finding edge costs..."
	fill_costs(pieces)
"""
	std::vector<match_score>::iterator i= matches.begin();
	PuzzleDisjointSet p((int)pieces.size());

	int output_id=0;
	while(!p.in_one_set() && i!=matches.end() ){
		int p1 = i->edge1/4;
		int e1 = i->edge1%4;
		int p2 = i->edge2/4;
		int e2 = i->edge2%4;
		
//Uncomment the following lines to spit out pictures of the matched edges...
//        cv::Mat m = cv::Mat::zeros(500,500,CV_8UC1);
//        std::stringstream out_file_name;
//        out_file_name << "/tmp/final/match" << output_id++ << "_" << p1<< "_" << e1 << "_" <<p2 << "_" <<e2 << ".png";
//        std::vector<std::vector<cv::Point> > contours;
//        contours.push_back(pieces[p1].edges[e1].get_translated_contour(200, 0));
//        contours.push_back(pieces[p2].edges[e2].get_translated_contour_reverse(200, 0));
//        cv::drawContours(m, contours, -1, cv::Scalar(255));
//        std::cout << out_file_name.str() << std::endl;
//        cv::imwrite(out_file_name.str(), m);
//        std::cout << "Attempting to merge: " << p1 << " with: " << p2 << " using edges:" << e1 << ", " << e2 << " c:" << i->score << " count: "  << output_id++ <<std::endl;
		p.join_sets(p1, p2, e1, e2);
		i++;
	}
	
	if(p.in_one_set()){
		std::cout << "Possible solution found" << std::endl;
		solved = true;
		solution = p.get(p.find(1)).locations;
		solution_rotations = p.get(p.find(1)).rotations;
		
		for(int i =0; i<solution.size[0]; i++){
			for(int j=0; j<solution.size[1]; j++){
				int piece_number = solution(i,j);
				pieces[piece_number].rotate(4-solution_rotations(i,j));
			}
		}
		
		
	}
	
	
	
}
"""