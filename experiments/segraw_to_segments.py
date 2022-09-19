import pickle
import sys
import os

def create_if_not_exists(path):
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ''):
        os.makedirs(directory)

def write_pkl(o, path):
    create_if_not_exists(path)
    with open(path, 'wb') as f:
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)

# convert from their custom format to pkl file of returned segments
# these segments fit into the existing process for this project
def segraw_to_segments(path):
	out_path = path.replace('segraw','pkl')

	lines = tuple(open(path, 'r'))

	all_segments = []
	this_group = []
	for l in lines:
		if 'pattern' in l:
			all_segments.append(this_group)
			this_group = []
		else:
			x0,y0,x1,y1 = l.replace('\n','').replace('-> ', ' ').replace('->', ' ').strip().split(' ')
			this_group.append(((int(x0),int(y0)),(int(x1), int(y1))))

	write_pkl(all_segments, out_path)

if __name__ == '__main__':
    path = sys.argv[1]
    segraw_to_sements(path)
