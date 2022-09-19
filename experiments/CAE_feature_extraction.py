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


for i_path in range(len(all_paths))[1:]:
    try:
        (title, raga, tonic), (file, mask_file, pitch_file) = all_paths[i_path]
        track_name = pitch_file.replace('.csv','').split('/')[-1]
        out_dir = f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{track_name}/'
        if False:#os.path.isfile(os.path.join(out_dir,'self_sim.npy')):
            print(f'Skipping {track_name}')
        else:
            print('\n---------')
            print(title)
            print('---------')

            #def create_matrices(file, mask_file, length_ngram, cuda, data_type, n_bins, bins_per_oct, fmin, hop_length, step_size=1, mode='cosine'):
            print('Computing CAE features')
            data = get_input_repr(file, data_type, n_bins, bins_per_oct, fmin, hop_length)
            mask, time, timestep  = get_timeseries(mask_file)
            pitch, fr_time, fr_timestep  = get_timeseries(pitch_file)

            ampls, phases = to_amp_phase(model, data, step_size=step_size, length_ngram=length_ngram, cuda=cuda)
            results = np.array([ampls, phases])

            print('Computing self similarity matrix')
            # get mask of silent and stable regions
            matrix_mask = []
            for i in range(ampls.shape[0]):
                t = (i+1)*hop_length/sr
                ix = find_nearest(time, t)
                matrix_mask.append(mask[ix])
            matrix_mask = np.array(matrix_mask)

            good_ix = np.where(matrix_mask==0)[0]
            orig_sparse_lookup = {g:s for s,g in enumerate(good_ix)}
            sparse_orig_lookup = {s:g for g,s in orig_sparse_lookup.items()}
            boundaries_orig = []
            for i in range(1, len(matrix_mask)):
                curr = matrix_mask[i]
                prev = matrix_mask[i-1]
                if curr==0 and prev==1:
                    boundaries_orig.append(i)
                elif curr==1 and prev==0:
                    boundaries_orig.append(i-1)
            boundaries_sparse = np.array([orig_sparse_lookup[i] for i in boundaries_orig])
            # boundaries contain two consecutive boundaries for each gap
            # but not if the excluded region leads to the end of the track
            red_boundaries_sparse = []
            boundaries_mask = [0]*len(boundaries_sparse)
            for i in range(len(boundaries_sparse)):
                if i==0:
                    red_boundaries_sparse.append(boundaries_sparse[i])
                    boundaries_mask[i]=1
                if boundaries_mask[i]==1:
                    continue
                curr = boundaries_sparse[i]
                prev = boundaries_sparse[i-1]
                if curr-prev == 1:
                    red_boundaries_sparse.append(prev)
                    boundaries_mask[i]=1
                    boundaries_mask[i-1]=1
                else:
                    red_boundaries_sparse.append(curr)
                    boundaries_mask[i]=1
            boundaries_sparse = np.array(sorted(list(set(red_boundaries_sparse))))

            sparse_ampls = ampls[good_ix]
            matrix_orig = create_ss_matrix(sparse_ampls, mode=mode)

            print('Normalising self similarity matrix')
            matrix = 1 / (matrix_orig + 1e-6)

            for k in range(-8, 9):
                eye = 1 - np.eye(*matrix.shape, k=k)
                matrix = matrix * eye

            flength = 10
            ey = np.eye(flength) + np.eye(flength, k=1) + np.eye(flength, k=-1)
            matrix = convolve2d(matrix, ey, mode="same")

            diag_mask = np.ones(matrix.shape)
            diag_mask = (diag_mask - np.diag(np.ones(matrix.shape[0]))).astype(np.bool)

            mat_min = np.min(matrix[diag_mask])
            mat_max = np.max(matrix[diag_mask])

            matrix -= matrix.min()
            matrix /= (matrix.max() + 1e-8)

            #for b in boundaries_sparse:
            #    matrix[:,b] = 1
            #    matrix[b,:] = 1

            matrix[~diag_mask] = 0

            #plt.imsave('random/4final.png', matrix, cmap="hot")

            ## Output
            metadata = {
                'orig_size': (len(data), len(data)),
                'sparse_size': (matrix.shape[0], matrix.shape[0]),
                'orig_sparse_lookup': orig_sparse_lookup,
                'sparse_orig_lookup': sparse_orig_lookup,
                'boundaries_orig': boundaries_orig,
                'boundaries_sparse': boundaries_sparse,
                'audio_path': file,
                'pitch_path': pitch_file,
                'stability_path': mask_file,
                'raga': raga,
                'tonic': tonic,
                'title': title
            }

            out_path_mat = os.path.join(out_dir, 'self_sim.npy')
            out_path_meta = os.path.join(out_dir, 'metadata.pkl')
            out_path_feat = os.path.join(out_dir, "features.pyc.bz")

            create_if_not_exists(out_dir)

            print(f"Saving features to {out_path_feat}..")
            save_pyc_bz(results, out_path_feat)

            print(f"Saving self sim matrix to {out_path_mat}..")
            np.save(out_path_mat, matrix)

            print(f'Saving metadata to {out_path_meta}')
            write_pkl(metadata, out_path_meta)
    except Exception as e:
        print(f'{title} failed')
        print(f'{e}')


