from pathlib import Path
import os
import shutil
import tqdm

from src.io import create_if_not_exists

def get_all_paths(d):
    p = Path(d)
    paths = [str(x).replace(d,'') for x in p.glob('**/*') if x.is_file()]
    return paths

dataset_dir = '/Volumes/Shruti/data/compmusic/CarnaticCC/audio/'
dest_dir = '/Volumes/Shruti/data/compmusic/CarnaticCC_clean/audio/'


paths = get_all_paths(dataset_dir)

paths = [x for x in paths if 'multitrack' not in x and 'DS_Store' not in x]

for p in tqdm.tqdm(paths):
	dest = os.path.join(dest_dir, p)
	create_if_not_exists(dest)
	shutil.copyfile(os.path.join(dataset_dir, p), dest)