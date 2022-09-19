import os

from pathlib import Path

def get_all_paths(d, ext='.mp3'):
    p = Path(d)
    paths = [str(x).replace(d,'') for x in p.glob('**/*') if ext in str(x)]
    return paths

dataset_dir = '/homedtic/tnuttall/asplab2/cae-invar/audio/compmusic/data/Carnatic/'

all_paths = get_all_paths(dataset_dir, '.mp3')
with open('carnatic_filelist.txt', 'w') as f:
    for p in all_paths:
        t = os.path.join(dataset_dir, p)
        f.write(f'{t}\n')

import os

from pathlib import Path

def get_all_paths(d, ext='.mp3'):
    p = Path(d)
    paths = [str(x).replace(d,'') for x in p.glob('**/*') if ext in str(x)]
    return paths

dataset_dir = '/homedtic/tnuttall/asplab2/cae-invar/audio/compmusic/data/Carnatic_vocal/'

all_paths = get_all_paths(dataset_dir, '.mp3')
with open('carnatic_vocal_filelist.txt', 'w') as f:
    for p in all_paths:
        t = os.path.join(dataset_dir, p)
        f.write(f'{t}\n')




import os

from pathlib import Path

def get_all_paths(d, ext='.mp3'):
    p = Path(d)
    paths = [str(x).replace(d,'') for x in p.glob('**/*') if ext in str(x)]
    return paths

dataset_dir = '/homedtic/tnuttall/asplab2/cae-invar/audio/compmusic/data/Carnatic_vocal/'

all_paths = get_all_paths(dataset_dir, 'multitrack-vocal.mp3')
with open('carnatic_multitrack.txt', 'w') as f:
    for p in all_paths:
        t = os.path.join(dataset_dir, p)
        f.write(f'{t}\n')

