
%load_ext autoreload
%autoreload 2

import skimage.io

from convert import *
from src.pitch import *
from src.io import *
from all_paths import all_paths

run_keyword= 'hpc'

cache_dir = "cache"
cuda = False
train_it = True
continue_train = False
start_epoch = 0
test_data = 'jku_input.txt'
test_cqt = False

data_type = 'cqt'

# train params
samples_epoch = 100000
batch_size = 1000
epochs = 1000
lr = 1e-3
sparsity_reg = 0e-5
weight_reg = 0e-5
norm_loss = 0
equal_norm_loss = 0
learn_norm = False
set_to_norm = -1
power_loss = 1
seed = 1
plot_interval = 500

# model params
dropout = 0.5
n_bases = 256

# CQT params
length_ngram = 32
fmin = 65.4
hop_length = 1984
block_size = 2556416
n_bins = 120
bins_per_oct = 24
sr = 44100

# MIDI params
min_pitch = 40
max_pitch = 100
beats_per_timestep = 0.25

# data loader
emph_onset = 0
rebuild = False
# shiftx (time), shifty (pitch)
shifts = 12, 24
# scalex, scaley (scaley not implemented!)
scales = 0, 0
# transform: 0 -> shifty (pitch), 1 -> shiftx (time), 2 -> scalex (time)
transform = 0, 1


torch.manual_seed(seed)
np.random.seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
out_dir = os.path.join("output", run_keyword)

assert os.path.exists(out_dir), f"The output directory {out_dir} does " \
    f"not exist. Did you forget to train using the run_keyword " \
    f"{run_keyword}?"

if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

if data_type == 'cqt':
    in_size = n_bins * length_ngram
else:
    raise AttributeError(f"Data_type {data_type} not supported. "
                         f"Possible type is 'cqt'.")
model = Complex(in_size, n_bases, dropout=dropout)
model_save_fn = os.path.join(out_dir, "model_complex_auto_"
                                     f"{data_type}.save")
model.load_state_dict(torch.load(model_save_fn, map_location='cpu'), strict=False)


def find_nearest(array, value, ix=True):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if ix:
        return idx
    else:
        return array[idx]


# params from config
length_ngram=32
cuda=False
data_type = 'cqt'
n_bins=120
bins_per_oct=24
fmin=65.4
hop_length=1984
step_size=1
mode='cosine'



audios = ['../for_genis_spleeter/Rithvik Raja - Emani Migula Spleeter.mp3',
'../for_genis_spleeter/Sanjay Subrahmanyan - Entara Nitana Spleeter.mp3',
'../for_genis_spleeter/Sanjay Subrahmanyan - Eranapai Spleeter.mp3',
'../for_genis_spleeter/Sanjay Subrahmanyan - Kamakshi Spleeter.mp3']


for aud in audios:

        track_name = aud.replace('.mp3','').split('/')[-1]
        out_dir = f'/Volumes/Shruti/asplab2/for_genis_features/'

        out_path_feat = os.path.join(out_dir, f"{track_name}_features.pyc.bz")

        create_if_not_exists(out_dir)

        print('Computing CAE features')
        data = get_input_repr(aud, data_type, n_bins, bins_per_oct, fmin, hop_length)
        ampls, phases = to_amp_phase(model, data, step_size=step_size, length_ngram=length_ngram, cuda=cuda)
        results = np.array([ampls, phases])

        print(f"Saving features to {out_path_feat}..")
        save_pyc_bz(results, out_path_feat)


