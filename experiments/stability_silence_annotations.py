#%load_ext autoreload
#%autoreload 2
import librosa
import sys
import datetime
import os
import pickle

import numpy as np
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


total = 0
contain_silence = 0
contain_stability = 0
contain_silstab = 0
start_silence = 0
end_silence = 0
start_stability = 0
end_stability = 0
start_stab_silence = 0
end_stab_silence = 0
enclosed_by_silstab = 0
bad_patterns = [[],[],[]]
all_annotations = []
all_pitch = []
all_time = []
for j,track_name in enumerate(['Koti Janmani', 'Vanajaksha Ninne Kori', 'Sharanu Janakana']):

    # pitch track extraction
    gap_interp = 250*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

    # stability identification
    stab_hop_secs = 0.2 # window size for stab computations in seconds
    min_stability_length_secs = 1.0 # minimum length of region to be considered stable in seconds
    freq_var_thresh_stab = 70 # max variation in pitch (in cents) to be considered stable region

    sr = 44100

    lc_name = '_'.join([x.lower() for x in track_name.split(' ')])
    annotations_path = f'annotations/{lc_name}.txt'
    out_dir = os.path.join(f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{track_name}/')
    metadata_path = os.path.join(out_dir, 'metadata.pkl')

    print(f'Loading metadata from {metadata_path}')
    metadata = load_pkl(metadata_path)

    raga = metadata['raga']
    print(f'Raga: {raga}')

    ## Unpack Metadata
    audio_path = metadata['audio_path']
    pitch_path = metadata['pitch_path']
    stability_path = metadata['stability_path']
    raga = metadata['raga']
    tonic = metadata['tonic']

    print('Loading pitch track')
    raw_pitch, time, timestep = get_timeseries(pitch_path)
    raw_pitch[np.where(raw_pitch<80)[0]]=0
    raw_pitch = interpolate_below_length(raw_pitch, 0, (gap_interp/timestep))
    tot_l = len(raw_pitch)*timestep

    print('Computing stability/silence mask')
    seg_hash = str((min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab))
    seg_path = os.path.join(out_dir, f'stability/{seg_hash}.pkl')
    raw_pitch_cents = pitch_seq_to_cents(raw_pitch, tonic)
    stable_mask = get_stability_mask(raw_pitch_cents, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep)
    silence_mask = (raw_pitch == 0).astype(int)
    silence_mask = add_center_to_mask(silence_mask)
    silence_and_stable_mask = np.array([int(any([i,j])) for i,j in zip(silence_mask, stable_mask)])

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

    audio, _ = librosa.load(audio_path)
    annotations = load_annotations_brindha(annotations_path, min_m=1.5)
    all_annotations.append(annotations)
    all_pitch.append(raw_pitch)
    all_time.append(time)
    print('\n\n\n')
    print(track_name)
    for i in range(len(annotations)):

        total += 1

        ann = annotations.iloc[i]

        s1 = round(ann.s1/timestep)
        s2 = round(ann.s2/timestep)
        
        if (not 1 in silence_and_stable_mask[s1-20:s1+20]) and (not 1 in silence_and_stable_mask[s2-20:s2+20]) and (1 in silence_and_stable_mask[s1+21:s2-21]):
            bad_patterns[j].append(i)
            contain_silstab += 1

        if 1 in silence_and_stable_mask[s1-20:s1+20] or 1 in silence_and_stable_mask[s2-20:s2+20]:
            enclosed_by_silstab += 1


print(f"Proportion of annotations that that contain silence/stability: {round(contain_silstab*100/total, 2)}%")
print(f"Proportion of annotations that are enclosed by silence/stability: {round(enclosed_by_silstab*100/total, 2)}%")


from src.visualisation import plot_subsequence

for j,track in enumerate(bad_patterns):
    for i in track:
        track_name = ['Koti Janmani', 'Vanajaksha Ninne Kori', 'Sharanu Janakana'][j]

        ann = all_annotations[j].iloc[i]
        this_pitch = all_pitch[j]
        this_time = all_time[j]

        s1 = round(ann.s1/timestep)
        s2 = round(ann.s2/timestep)
        create_if_not_exists(f'bad_patterns/{track_name}/{i}.png')
        plot_subsequence(s1, s2-s1, this_pitch, this_time, timestep, path=f'bad_patterns/{track_name}/{i}.png', plot_kwargs=plot_kwargs)
