# annotations and segments plot
# annotations and returned pattern plot
# stability track and pitch plot
import numpy as np
import matplotlib.pyplot as plt


######### STABILITY #################
def pitch_stability(pitch, time, mask, t1, t2, path):
    this_pitch = pitch[t1:t2]
    this_time = time[t1:t2]
    this_mask = mask[t1:t2]
    
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(111)

    #fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Pitch (Hz)')
    ax1.plot(this_time, this_pitch, color=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Stability')  # we already handled the x-label with ax1
    ax2.plot(this_time, this_mask, color=color)

    fig.tight_layout()  # otherw ise the right y-label is slightly clipped
    plt.savefig(path)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 1.0 # minimum legnth of region to be considered stable in seconds
freq_var_thresh_stab = 8 # max variation in pitch to be considered stable region
stable_mask = get_stability_mask(raw_pitch, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep)


path = 'random/stab_check.png'
t1 = 10000
t2 = t1+10000
pitch_stability(raw_pitch, time, stable_mask, t1, t2, path)


########## ANNOTATION COMPARISON #####################
from src.visualisation import add_line_to_plot, get_lines, add_annotations_to_plot, add_patterns_to_plot, add_segments_to_plot, join_plots
import matplotlib

def plot_annotations_and_X(X, annotations, s1, timestep, sr, cqt_window):
    dl = s1*cqt_window/sr
    annotations1 = annotations.copy()
    annotations1['s1'] = annotations1['s1']-dl
    annotations1['s2'] = annotations1['s2']-dl
    annotations1 = annotations1[annotations1['s2']<= X.shape[0]*cqt_window/sr]


    # Annotations
    X_annotate = add_annotations_to_plot(X, annotations1, sr, cqt_window)
    skimage.io.imsave(path, X_annotate.astype(np.uint8))
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def plot_segments_orig(segments, s1, S, path):
    scale_segments = []
    for ((x0,y0), (x1, y1)) in segments:
        scale_segments.append([(x0-s1, y0-s1), (x1-s1, y1-s1)])
    
    X_canvas = np.zeros((S,S))
    # Found segments from image processing
    X_segments = add_segments_to_plot(X_canvas, scale_segments)

    skimage.io.imsave(path, X_segments.astype(np.uint8))
    
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def plot_segments_sparse(segments, S, path):

    X_canvas = np.zeros((S,S))
    # Found segments from image processing
    X_segments = add_segments_to_plot(X_canvas, segments)
    
    skimage.io.imsave(path, X_segments.astype(np.uint8))
    
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def plot_patterns(starts, lengths, S, s1, path):
    start_scaled = []
    for i in range(len(starts)):
        this_start = starts[i]
        new_start = []
        for j in range(len(this_start)):
            s = this_start[j]
            new_start.append(s-s1)
        start_scaled.append(new_start)

    X_canvas = np.zeros((S,S))

    # Patterns from full pipeline
    X_patterns = add_patterns_to_plot(X_canvas, start_scaled, lengths, sr, cqt_window)
    skimage.io.imsave(path, X_patterns.astype(np.uint8))


if save_img:
    S = sparse_orig_lookup[s2] - sparse_orig_lookup[s1]

    path = os.path.join(out_dir, 'progress_plots/A_annotations.png')
    plot_annotations_and_X(X_samp, annotations_raw, sparse_orig_lookup[s1], timestep, sr, cqt_window)

    path = os.path.join(out_dir, 'progress_plots/B_segments_extended.png')
    plot_segments_sparse(all_segments_extended_reduced, s2-s1, path)

    path = os.path.join(out_dir, 'progress_plots/C_segments_scaled_reduced.png')
    plot_segments_sparse(all_segments_scaled_reduced, s2-s1, path)

    path = os.path.join(out_dir, 'progress_plots/D_segments_converted.png')
    plot_segments_orig(all_segments_converted, sparse_orig_lookup[s1], S, path)

    path = os.path.join(out_dir, 'progress_plots/E_segments_reduced.png')
    plot_segments_orig(all_segments_reduced, sparse_orig_lookup[s1], S, path)

    path = os.path.join(out_dir, 'progress_plots/F_final_patterns.png')
    plot_patterns(starts_sec, lengths_sec, S, sparse_orig_lookup[s1]*cqt_window/sr, path)


