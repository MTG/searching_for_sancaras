import os

import numpy as np
import pandas as pd
import skimage.io

from src.img import (
    remove_diagonal, convolve_array_tile, binarize, diagonal_gaussian, 
    sobel, apply_bin_op, make_symmetric, edges_to_contours)
from src.segments import (
    extract_segments_new, break_all_segments, remove_short, extend_segments, 
    join_all_segments, extend_groups_to_mask, group_segments, group_overlapping, 
    group_by_distance, trim_silence, sparse_to_orig)
from src.sequence import (
    convert_seqs_to_timestep, get_stability_mask, 
    add_center_to_mask, remove_below_length)
from src.evaluation import evaluate, load_annotations_new, get_coverage, get_grouping_accuracy
from src.visualisation import plot_all_sequences, flush_matplotlib
from src.io import load_sim_matrix, write_all_sequence_audio, load_yaml, load_pkl, create_if_not_exists, write_pkl, run_or_cache
from src.pitch import cents_to_pitch, get_timeseries, interpolate_below_length

# Sample rate of audio
sr = 44100

# size in frames of cqt window from convolution model
cqt_window = 1984 # was previously set to 1988

# Take sample of data, set to None to use all data
s1 =  None # lower bound index (5000 has been used for testing)
s2 =  None # higher bound index (9000 has been used for testing)

# pitch track extraction
gap_interp = 0.25 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 1.0 # minimum legnth of region to be considered stable in seconds
freq_var_thresh_stab = 10 # max variation in pitch to be considered stable region

conv_filter_str = 'sobel'

# Binarize raw sim array 0/1 below and above this value...
# depends completely on filter passed to convolutional step
# Best...
#   scharr, 0.56
#   sobel unidrectional, 0.1
#   sobel bidirectional, 0.15
bin_thresh = 0.16
# lower bin_thresh for areas surrounding segments
bin_thresh_segment = 0.08
# percentage either size of a segment considered for lower bin thresh
perc_tail = 0.5

# Gaussian filter along diagonals with sigma...
gauss_sigma = None

# After gaussian, re-binarize with this thresholdth
cont_thresh = 0.15

# morphology params
etc_kernel_size = 10 # For closing
binop_dim = 3 # square dimension of binary opening structure (square matrix of zeros with 1 across the diagonal)

# Distance between consecutive diagonals to be joined in seconds
min_diff_trav = 0.5 #0.1

# Grouping diagonals
min_pattern_length_seconds = 2
min_in_group = 1 # minimum number of patterns to be included in pattern group

# Joining groups
match_tol = 1 # 22==1 seconds
# extend to silent/stability mask using this proportion of pattern
ext_mask_tol = 0.5 

n_dtw = 10 # number of samples to take from each group to compare dtw values
thresh_dtw = 3.3 # if average dtw distance (per sequence) between two groups is below this threshold, group them
thresh_cos =  None # if average cos distance (per sequence) between two groups is below this threshold, group them
# Two segments must overlap in both x and y by <dupl_perc_overlap> 
# to be considered the same, only the longest is returned
dupl_perc_overlap = 0.95

# evaluation
partial_perc = 0.66 # how much overlap does an annotated and identified pattern needed to be considered a partial match

# limit the number of groups outputted
top_n = 1000

# output
write_plots = True
write_audio = True
write_patterns = True
write_annotations = True
plot = False
run_name = 'pipeline_test'

segment_paths = [
    'output/for_paper/from_original_cae/Vanajaksha Ninne Kori Spleeter.pkl',
    'output/for_paper/from_original_cae/Koti Janmani Spleeter.pkl',
    'output/for_paper/from_original_cae/Sharanu Janakana Spleeter.pkl'
]
annotations_paths = [
    'annotations/vanajaksha_ninne_kori.txt',
    'annotations/koti_janmani.txt',
    'annotations/sharanu_janakana.txt'
]
track_names = [
    'Vanajaksha Ninne Kori',
    'Koti Janmani',
    'Sharanu Janakana'
]
#all_annotations = pd.DataFrame()

def run_pattern(i):
    track_name = track_names[i]
    path = segment_paths[i]
    all_segments = load_pkl(path)
    annotations_path = annotations_paths[i]

    ## Get Data
    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{track_name}/')
    metadata_path = os.path.join(out_dir, 'metadata.pkl')
    sim_path = os.path.join(out_dir, 'self_sim.npy')

    print(f'Loading metadata from {metadata_path}')
    metadata = load_pkl(metadata_path)
    arr_sparse_size = metadata['sparse_size']
    raga = metadata['raga']
    print(f'Raga: {raga}')
    print(f'Sparse array size: {arr_sparse_size}')

    audio_path = metadata['audio_path']
    pitch_path = metadata['pitch_path']
    raga = metadata['raga']
    tonic = metadata['tonic']

    print('Loading pitch track')
    raw_pitch, time, timestep = get_timeseries(pitch_path)
    raw_pitch[np.where(raw_pitch<80)[0]]=0
    raw_pitch = interpolate_below_length(raw_pitch, 0, (gap_interp/timestep))

    ### Image Processing
    # convolutional filter
    if conv_filter_str == 'sobel':
        conv_filter = sobel

    min_diff_trav_hyp = (2*min_diff_trav**2)**0.5 # translate min_diff_trav to corresponding diagonal distance
    min_diff_trav_seq = min_diff_trav_hyp*sr/cqt_window

    min_length_cqt = min_pattern_length_seconds*sr/cqt_window

    # Output
    svara_cent_path = "conf/svara_cents.yaml"
    svara_freq_path = "conf/svara_lookup.yaml"

    svara_cent = load_yaml(svara_cent_path)
    svara_freq = load_yaml(svara_freq_path)

    if raga in svara_freq:
        arohana = svara_freq[raga]['arohana']
        avorahana = svara_freq[raga]['avorahana']
        all_svaras = list(set(arohana+avorahana))
        print(f'Svaras for raga, {raga}:')
        print(f'   arohana: {arohana}')
        print(f'   avorahana: {avorahana}')

        yticks_dict = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}
        yticks_dict = {k:v for k,v in yticks_dict.items() if any([x in k for x in all_svaras])}

        plot_kwargs = {
            'yticks_dict':yticks_dict,
            'cents':True,
            'tonic':tonic,
            'emphasize':['S', 'S ', 'S  ', ' S', '  S'],
            'figsize':(15,4)
        }
    else:
        plot_kwargs = {
            'yticks_dict':{},
            'cents':False,
            'tonic':None,
            'emphasize':[],
            'figsize':(15,4)
        }
        print(f'No svara information for raga, {raga}')

    all_groups = []
    for g in all_segments:
        this_group = []
        for ((x0,y0),(x1,y1)) in g:
            this_group += [(x0,x1), (y0,y1)]
        if len(this_group):
            all_groups.append(this_group)

    print('Convert sequences to pitch track timesteps')
    starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups, cqt_window, sr, timestep)

    print('Applying exclusion functions')
    starts_seq_exc,  lengths_seq_exc = remove_below_length(starts_seq, lengths_seq, timestep, min_pattern_length_seconds)

    starts = [p for p in starts_seq_exc if len(p)>=min_in_group]
    lengths = [p for p in lengths_seq_exc if len(p)>=min_in_group]

    starts_sec = [[x*timestep for x in p] for p in starts]
    lengths_sec = [[x*timestep for x in l] for l in lengths]

    results_dir = os.path.join(out_dir, f'results/baseline/{track_name}')

    n_patterns = sum([len(x) for x in starts])
    coverage = get_coverage(raw_pitch, starts, lengths)
    print(f'Number of Patterns: {n_patterns}')
    print(f'Number of Groups: {len(starts_sec)}')
    print(f'Coverage: {round(coverage,2)}')
    
    annotations_raw = load_annotations_new(annotations_path, min_pattern_length_seconds, None)

    if s1:
        start_time = (sparse_orig_lookup[s1]*cqt_window)/sr
        end_time = (sparse_orig_lookup[s2]*cqt_window)/sr
        annotations_raw = annotations_raw[
            (annotations_raw['s1']>=start_time) & 
            (annotations_raw['s2']<=end_time)]

    annotations_r = annotations_raw[['tier', 's1', 's2', 'text']]

    recall, precision, f1, annotations = evaluate(annotations_r, starts_sec, lengths_sec, partial_perc)

    return recall, precision, f1, annotations


