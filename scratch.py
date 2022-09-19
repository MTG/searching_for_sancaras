#%load_ext autoreload
#%autoreload 2
import sys
import datetime
import os
import pickle

import skimage.io

from src.pitch import extract_pitch_track
from src.img import (
    remove_diagonal, convolve_array_tile, binarize, diagonal_gaussian, 
    hough_transform, hough_transform_new, scharr, sobel,
    apply_bin_op, make_symmetric, edges_to_contours)
from src.segments import (
    extract_segments_new, get_all_segments, break_all_segments, do_patterns_overlap, reduce_duplicates, 
    remove_short, extend_segments, join_all_segments, extend_groups_to_mask, group_segments, group_overlapping,
    group_by_distance, trim_silence, sparse_to_orig)
from src.sequence import (
    apply_exclusions, contains_silence, min_gap, too_stable, 
    convert_seqs_to_timestep, get_stability_mask, add_center_to_mask,
    remove_below_length, extend_to_mask, add_border_to_mask)
from src.evaluation import evaluate, load_annotations_brindha, get_coverage, get_grouping_accuracy
from src.visualisation import plot_all_sequences, plot_pitch, flush_matplotlib
from src.io import load_sim_matrix, write_all_sequence_audio, load_yaml, load_pkl, create_if_not_exists, write_pkl, run_or_cache
from src.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents, get_timeseries, interpolate_below_length

### What do we have?
# Track Names...
#   all_tracks = [
#       'Koti Janmani', 
#       'Karunimpa Idi',
#       'Sharanu Janakana',
#       'Siddhi Vinayakam',
#       'Vanajaksha Ninne Kori',
#       'Sundari Nee Divya',
#       'Shankari Shankuru',
#       'Karuna Nidhi Illalo', 
#       'Koluvamaregatha', 
#       'Mati Matiki', 
#       'Ramabhi Rama Manasu',
#       'Palisomma Muddu Sarade',
#       'Rama Rama Guna Seema',
#       'Lokavana Chatura']
all_tracks = [
    'Koti Janmani', 
    'Vanajaksha Ninne Kori',
    'Sharanu Janakana',
    'Sundari Nee Divya'
]
#for t in all_tracks:
#    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{t}/')
#    metadata_path = os.path.join(out_dir, 'metadata.pkl')
#    metadata = load_pkl(metadata_path)
#    title = t
#    raga = metadata['raga']
#    line = '-'*len(title)
#    ass = metadata['sparse_size']
#    print(f'{title}')
#    if title:
#        print(f'{line}')
#    print(f'    Raga: {raga}')
#    print(f'    Sparse array size: ({ass}, {ass})')

################
## Parameters ##
################
track_name = 'Sundari Nee Divya'
track_name = 'Sharanu Janakana'

# Sample rate of audio
sr = 44100

# size in frames of cqt window from convolution model
cqt_window = 1984 # was previously set to 1988

# Take sample of data, set to None to use all data
s1 = None # lower bound index (5000 has been used for testing)
s2 = None # higher bound index (9000 has been used for testing)

# pitch track extraction
gap_interp = 350*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 0.5 # minimum legnth of region to be considered stable in seconds
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

# After gaussian, re-binarize with this threshold
cont_thresh = 0.15

# morphology params
etc_kernel_size = 10 # For closing
binop_dim = 3 # square dimension of binary opening structure (square matrix of zeros with 1 across the diagonal)

# Distance between consecutive diagonals to be joined in seconds
min_diff_trav = 0.5 # 0.1

# Grouping diagonals
min_pattern_length_seconds = 2
min_in_group = 1 # minimum number of patterns to be included in pattern group

# Joining groups
match_tol = 1 # 22==1 seconds
# extend to silent/stability mask using this proportion of pattern
ext_mask_tol = 0.5 

n_dtw = 10 # number of samples to take from each group to compare dtw values
thresh_dtw = 3.3 # if average dtw distance (per sequence) between two groups is below this threshold, group them
thresh_cos = None  # if average cos distance (per sequence) between two groups is below this threshold, group them
# Two segments must overlap in both x and y by <dupl_perc_overlap> 
# to be considered the same, only the longest is returned
dupl_perc_overlap = 0.95

#%load_ext autoreload
#%autoreload 2
import sys
import datetime
import os
import pickle

import skimage.io


from src.pitch import extract_pitch_track
from src.img import (
    remove_diagonal, convolve_array_tile, binarize, diagonal_gaussian, 
    hough_transform, hough_transform_new, scharr, sobel,
    apply_bin_op, make_symmetric, edges_to_contours)
from src.segments import (
    extract_segments_new, get_all_segments, break_all_segments, do_patterns_overlap, reduce_duplicates, 
    remove_short, extend_segments, join_all_segments, extend_groups_to_mask, group_segments, group_overlapping,
    group_by_distance, trim_silence)
from src.sequence import (
    apply_exclusions, contains_silence, min_gap, too_stable, 
    convert_seqs_to_timestep, get_stability_mask, add_center_to_mask,
    remove_below_length, extend_to_mask, add_border_to_mask)
from src.evaluation import evaluate, load_annotations_brindha, get_coverage, get_grouping_accuracy
from src.visualisation import plot_all_sequences, plot_pitch, flush_matplotlib
from src.io import load_sim_matrix, write_all_sequence_audio, load_yaml, load_pkl, create_if_not_exists, write_pkl, run_or_cache
from src.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents, get_timeseries, interpolate_below_length

### What do we have?
# Track Names...
#   all_tracks = [
#       'Koti Janmani', 
#       'Karunimpa Idi',
#       'Sharanu Janakana',
#       'Siddhi Vinayakam',
#       'Vanajaksha Ninne Kori',
#       'Sundari Nee Divya',
#       'Shankari Shankuru',
#       'Karuna Nidhi Illalo', 
#       'Koluvamaregatha', 
#       'Mati Matiki', 
#       'Ramabhi Rama Manasu',
#       'Palisomma Muddu Sarade',
#       'Rama Rama Guna Seema',
#       'Lokavana Chatura']
all_tracks = [
    'Koti Janmani', 
    'Vanajaksha Ninne Kori',
    'Sharanu Janakana',
    'Sundari Nee Divya'
]
#for t in all_tracks:
#    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{t}/')
#    metadata_path = os.path.join(out_dir, 'metadata.pkl')
#    metadata = load_pkl(metadata_path)
#    title = t
#    raga = metadata['raga']
#    line = '-'*len(title)
#    ass = metadata['sparse_size']
#    print(f'{title}')
#    if title:
#        print(f'{line}')
#    print(f'    Raga: {raga}')
#    print(f'    Sparse array size: ({ass}, {ass})')

################
## Parameters ##
################
track_name = 'Sundari Nee Divya'
track_name = 'Sharanu Janakana'

# Sample rate of audio
sr = 44100

# size in frames of cqt window from convolution model
cqt_window = 1984 # was previously set to 1988

# Take sample of data, set to None to use all data
s1 = None # lower bound index (5000 has been used for testing)
s2 = None # higher bound index (9000 has been used for testing)

# pitch track extraction
gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

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

# After gaussian, re-binarize with this threshold
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
thresh_cos = None  # if average cos distance (per sequence) between two groups is below this threshold, group them
# Two segments must overlap in both x and y by <dupl_perc_overlap> 
# to be considered the same, only the longest is returned
dupl_perc_overlap = 0.95

# Exclusions
exclusion_functions = [contains_silence]

# evaluation
all_annotations_paths = [
    'annotations/koti_janmani.txt',
    'annotations/vanajaksha_ninne_kori.txt',
    'annotations/sharanu_janakana.txt',
    'annotations/sundari_nee_divya.txt'
]
all_alap_annotations_paths = [
    'annotations/koti_janmani_alap.tsv',
    'annotations/vanajaksha_ninne_kori.txt',
    'annotations/sharanu_janakana_alap.tsv',
    'annotations/sundari_nee_divya_alap.tsv'
]
annotations_path = 'annotations/sharanu_janakana.txt'

partial_perc = 0.66 # how much overlap does an annotated and identified pattern needed to be considered a partial match

# limit the number of groups outputted
top_n = 1000

write_plots = True
write_audio = True
write_patterns = True
write_annotations = True
plot=False
run_name = 'final'

def main(
    track_name, run_name, sr, cqt_window, s1, s2,
    gap_interp, stab_hop_secs, min_stability_length_secs, 
    freq_var_thresh_stab, conv_filter_str, bin_thresh,
    bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
    etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
    min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos,
    dupl_perc_overlap, exclusion_functions,
    annotations_path, partial_perc, top_n, write_plots, 
    write_audio, write_patterns, write_annotations, plot=False):
    if write_annotations and not annotations_path:
        print('WARNING: write_annotations==True but no annotations path has been passed, annotations will not be written')
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

    print(f'Loading self similarity from {sim_path}')
    X = load_sim_matrix(sim_path)

    ## Unpack Metadata
    arr_orig_size = metadata['orig_size']
    arr_sparse_size = metadata['sparse_size']
    orig_sparse_lookup = metadata['orig_sparse_lookup']
    sparse_orig_lookup = {v:k for k,v in orig_sparse_lookup.items()}
    sparse_orig_lookup[X.shape[0]] = sparse_orig_lookup[X.shape[0]-1]
    boundaries_orig = metadata['boundaries_orig']
    if s1 is not None:
        boundaries_sparse = [x-s1 for x in metadata['boundaries_sparse']]
    else:
        boundaries_sparse = metadata['boundaries_sparse']
    audio_path = metadata['audio_path']
    pitch_path = metadata['pitch_path']
    stability_path = metadata['stability_path']
    raga = metadata['raga']
    tonic = metadata['tonic']

    print('Loading pitch track')
    raw_pitch, time, timestep = get_timeseries(pitch_path)
    raw_pitch[np.where(raw_pitch<80)[0]]=0
    raw_pitch = interpolate_below_length(raw_pitch, 0, (gap_interp/timestep))

    print('Computing stability/silence mask')
    seg_hash = str((min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab))
    seg_path = os.path.join(out_dir, f'stability/{seg_hash}.pkl')
    stable_mask = run_or_cache(get_stability_mask, [raw_pitch, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep], seg_path)
    stable_mask = get_stability_mask(raw_pitch, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep)
    silence_mask = (raw_pitch == 0).astype(int)
    silence_mask = add_center_to_mask(silence_mask)
    silence_and_stable_mask = np.array([int(any([i,j])) for i,j in zip(silence_mask, stable_mask)])

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

    ####################
    ## Load sim array ##
    ####################
    # Sample for development
    if all([s1 is not None, s2 is not None]):
        save_imgs = s2-s1 <= 4000
        X_samp = X.copy()[s1:s2,s1:s2]
    else:
        save_imgs = False
        X_samp = X.copy()

    if not plot:
        save_imgs = False
    
    sim_filename = os.path.join(out_dir, 'progress_plots', '1_simsave.png') if save_imgs else None
    conv_filename = os.path.join(out_dir, 'progress_plots', '2_conv.png') if save_imgs else None
    bin_filename = os.path.join(out_dir, 'progress_plots', '3_binary.png') if save_imgs else None
    diag_filename = os.path.join(out_dir, 'progress_plots', '4_diag.png') if save_imgs else None
    gauss_filename = os.path.join(out_dir, 'progress_plots', '5_gauss.png') if save_imgs else None
    cont_filename = os.path.join(out_dir, 'progress_plots', '6_cont.png') if save_imgs else None
    close_filename = os.path.join(out_dir, 'progress_plots', '6_close.png') if save_imgs else None
    binop_filename = os.path.join(out_dir, 'progress_plots', '7_binop.png') if save_imgs else None
    hough_filename = os.path.join(out_dir, 'progress_plots', '8_hough.png') if save_imgs else None
    ext_filename = os.path.join(out_dir, 'progress_plots', '9_cont_ext.png') if save_imgs else None


    if save_imgs:
        create_if_not_exists(sim_filename)
        skimage.io.imsave(sim_filename, X_samp)

    ##############
    ## Pipeline ##
    ##############
    print('Convolving similarity matrix')
    # Hash all parameters used before segment finding to hash results later
    conv_hash = str((s1, s2, conv_filter_str))
    conv_path = os.path.join(out_dir, f'convolve/{conv_hash}.pkl')
    X_conv = run_or_cache(convolve_array_tile, [X_samp, conv_filter], conv_path)
    
    #X_conv = convolve_array_tile(X_samp, cfilter=conv_filter)

    if save_imgs:
        skimage.io.imsave(conv_filename, X_conv)

    print('Binarizing convolved array')
    X_bin = binarize(X_conv, bin_thresh, filename=bin_filename)
    #X_bin = binarize(X_conv, 0.05, filename=bin_filename)

    print('Removing diagonal')
    X_diag = remove_diagonal(X_bin)

    if save_imgs:
        skimage.io.imsave(diag_filename, X_diag)

    if gauss_sigma:
        print('Applying diagonal gaussian filter')
        diagonal_gaussian(X_bin, gauss_sigma, filename=gauss_filename)

        print('Binarize gaussian blurred similarity matrix')
        binarize(X_gauss, cont_thresh, filename=cont_filename)
    else:
        X_gauss = X_diag
        X_cont  = X_gauss

    print('Ensuring symmetry between upper and lower triangle in array')
    X_sym = make_symmetric(X_cont)

    print('Identifying and isolating regions between edges')
    X_fill = edges_to_contours(X_sym, etc_kernel_size)

    if save_imgs:
        skimage.io.imsave(close_filename, X_fill)

    print('Cleaning isolated non-directional regions using morphological opening')
    X_binop = apply_bin_op(X_fill, binop_dim)

    print('Ensuring symmetry between upper and lower triangle in array')
    X_binop = make_symmetric(X_binop)

    if save_imgs:
        skimage.io.imsave(binop_filename, X_binop)
    
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim
    ))

    segment_path = os.path.join(out_dir, f'segments/{seg_hash}.pkl')
    all_segments = run_or_cache(extract_segments_new, [X_binop], segment_path)

    print('Extending Segments')  
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim, perc_tail, bin_thresh_segment
    ))

    segment_path = os.path.join(out_dir, f'segments_extended/{seg_hash}.pkl')
    all_segments_extended = run_or_cache(extend_segments, [all_segments, X_sym, X_conv, perc_tail, bin_thresh_segment], segment_path)

    print(f'    {len(all_segments_extended)} extended segments...')

    from src.segments import line_through_points
    
    all_segments_extended_reduced = remove_short(all_segments_extended, 1)

    all_segments_converted = sparse_to_orig(all_segments_extended_reduced, boundaries_sparse, sparse_orig_lookup, s1)
    
    #   [(i,(get_grad(x),get_grad(y))) for i,(x,y) in enumerate(zip(all_segments_scaled_reduced, all_segments_converted)) if get_grad(x) != get_grad(y)]

    #   get_grad = lambda y: (y[1][1]-y[0][1])/(y[1][0]-y[0][0])
    #   def check_seg(seg,boundaries_sparse):
    #       ((x0, y0), (x1, y1)) = seg
    #       return any([[x for x in boundaries_sparse if x0 <= x <= x1], [y for y in boundaries_sparse if y0 <= y <= y1]])

    print('Joining segments that are sufficiently close')
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim, perc_tail, 
        bin_thresh_segment, min_diff_trav_seq
    ))

    segment_path = os.path.join(out_dir, f'segments_joined/{seg_hash}.pkl')

    all_segments_joined = run_or_cache(join_all_segments, [all_segments_converted, min_diff_trav_seq], segment_path)
    print(f'    {len(all_segments_joined)} joined segments...')

    print('Breaking segments with silent/stable regions')
    # Format - [[(x,y), (x1,y1)],...]
    all_broken_segments = break_all_segments(all_segments_joined, silence_mask, cqt_window, sr, timestep)
    all_broken_segments = break_all_segments(all_broken_segments, stable_mask, cqt_window, sr, timestep)
    print(f'    {len(all_broken_segments)} broken segments...')

    #[(i,((x0,y0), (x1,y1))) for i,((x0,y0), (x1,y1)) in enumerate(all_segments) if x1-x0>10000]
    print('Reducing Segments')
    all_segments_reduced = remove_short(all_broken_segments, min_length_cqt)
    print(f'    {len(all_segments_reduced)} segments above minimum length of {min_pattern_length_seconds}s...')


    print('Identifying Segment Groups')
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim, perc_tail, 
        bin_thresh_segment, min_diff_trav_seq, min_length_cqt, match_tol
    ))
    segment_path = os.path.join(out_dir, f'segments_groups/{seg_hash}.pkl')
    all_groups = run_or_cache(group_segments, [all_segments_reduced, min_length_cqt, match_tol, silence_and_stable_mask, cqt_window, timestep, sr], segment_path)

    #all_groups = group_segments(all_segments_reduced, min_length_cqt, match_tol, silence_and_stable_mask, cqt_window, timestep, sr)
    print(f'    {len(all_groups)} segment groups found...')

    print('Extending segments to silence/stability')
    silence_and_stable_mask_2 = np.array([1 if any([i==2, j==2]) else 0 for i,j in zip(silence_mask, stable_mask)])
    all_groups_ext = extend_groups_to_mask(all_groups, silence_and_stable_mask_2, cqt_window, sr, timestep, toler=ext_mask_tol)

    print('Trimming Silence')
    all_groups_sil = trim_silence(all_groups_ext, raw_pitch, cqt_window, sr, timestep)

    print('Joining Groups of overlapping sequences')
    all_groups_over = group_overlapping(all_groups_sil, dupl_perc_overlap)
    print(f'    {len(all_groups_over)} groups after join...')

    if thresh_dtw:
        print('Joining geometrically close groups using pitch tracks')
        all_groups_dtw = group_by_distance(all_groups_over, raw_pitch, n_dtw, thresh_dtw, thresh_cos, cqt_window, sr, timestep)
        print(f'    {len(all_groups_dtw)} groups after join...')
    else:
        all_groups_dtw = all_groups_over

    print('Convert sequences to pitch track timesteps')
    starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups_dtw, cqt_window, sr, timestep)

    print('Applying exclusion functions')
    starts_seq_exc,  lengths_seq_exc = remove_below_length(starts_seq, lengths_seq, timestep, min_pattern_length_seconds)

    starts = [p for p in starts_seq_exc if len(p)>=min_in_group]
    lengths = [p for p in lengths_seq_exc if len(p)>=min_in_group]

    starts_sec = [[x*timestep for x in p] for p in starts]
    lengths_sec = [[x*timestep for x in l] for l in lengths]

    ############
    ## Output ##
    ############
    results_dir = os.path.join(out_dir, f'results/{run_name}/')
    
    if save_imgs:
        S = sparse_orig_lookup[s2] - sparse_orig_lookup[s1]

        path = os.path.join(out_dir, 'progress_plots/A_annotations.png')
        plot_annotations_and_X(X_samp, annotations_raw, sparse_orig_lookup[s1], timestep, sr, cqt_window)

        path = os.path.join(out_dir, 'progress_plots/B_segments_extended.png')
        plot_segments_sparse(all_segments_extended, s2-s1, path)
        
        path = os.path.join(out_dir, 'progress_plots/C_segments_scaled_reduced.png')
        plot_segments_sparse(all_segments_scaled_reduced, s2-s1, path)

        path = os.path.join(out_dir, 'progress_plots/D_segments_converted.png')
        plot_segments_orig(all_segments_converted, sparse_orig_lookup[s1], S, path)

        path = os.path.join(out_dir, 'progress_plots/E_segments_reduced.png')
        plot_segments_orig(all_segments_reduced, sparse_orig_lookup[s1], S, path)

        path = os.path.join(out_dir, 'progress_plots/F_final_patterns.png')
        plot_patterns(starts_sec, lengths_sec, S, sparse_orig_lookup[s1]*cqt_window/sr, path)


    print('Writing all sequences')
    if write_plots:
        plot_all_sequences(raw_pitch, time, lengths[:top_n], starts[:top_n], results_dir, clear_dir=True, plot_kwargs=plot_kwargs)
    
    if write_audio:
        write_all_sequence_audio(audio_path, starts[:top_n], lengths[:top_n], timestep, results_dir)
    
    if write_patterns:
        write_pkl(lengths[:top_n], os.path.join(results_dir, 'lengths.pkl'))
        write_pkl(starts[:top_n], os.path.join(results_dir, 'starts.pkl'))

    flush_matplotlib()

    if annotations_path:
        return recall, precision, f1, grouping_accuracy, group_distribution, annotations, starts_sec, lengths_sec
    else:
        return None, None, None, None, None, None, None, None
