
from src.segments import *
print('Grouping Segments')


all_groups_ext = [remove_group_duplicates(g, 0.01) for g in all_groups_ext]
all_groups_ext = [x for x in all_groups_ext if len(x) > 0]


## Remove those that are identical
group_match_dict = {}
match_dtws = []
for i, ag1 in enumerate(all_groups_ext):
    for j, ag2 in enumerate(all_groups_ext):
        sg = same_group(ag1, ag2, 0.85)
        if sg:
            update_dict(group_match_dict, i, j)
            update_dict(group_match_dict, j, i)

all_groups_ix = matches_dict_to_groups(group_match_dict)
all_groups_ix = [list(set(x)) for x in all_groups_ix]

import random




pos_groups = [x for x in all_groups_ix if len(x) > 1]

to_seqs = lambda y: round((y*cqt_window)/(sr*timestep))

all_dtw_known = []
all_cos_known = []
for group in tqdm.tqdm(pos_groups):
    for i1 in tqdm.tqdm(group):
        for i2 in group:
            if i2 >= i1:
                continue
            ag1 = all_groups[i1]
            ag2 = all_groups[i2]
            sample1 = random.sample(ag1, min(10, len(ag1)))
            sample2 = random.sample(ag2, min(10, len(ag2)))
            this_cos = []
            this_dtw = []
            for (x0, x1), (y0, y1) in itertools.product(sample1, sample2):
                seq1 = np.trim_zeros(pitch[to_seqs(x0): to_seqs(x1)]  )
                seq2 = np.trim_zeros(pitch[to_seqs(y0): to_seqs(y1)])
                
                seq1_hist = np.histogram(seq1, 100, density=True, range=(r1, r2))[0]
                seq2_hist = np.histogram(seq2, 100, density=True, range=(r1, r2))[0]

                hist_dist = cosine(seq1_hist, seq2_hist)
                seq_len = min([len(seq1), len(seq2)])
                val,conn = fastdtw.fastdtw(seq1, seq2, radius=round(seq_len*0.5))
                dist = val/len(conn)
                if dist > 3:
                    this_dtw.append(val/len(conn))
                    this_cos.append(hist_dist)
            all_dtw_known.append(np.mean(this_dtw))
            all_cos_known.append(np.mean(this_cos))


plt.hist(all_dtw_known, 100)
plt.title('DTW between known groups')
plt.ylabel('DTW/sequence length')
plt.savefig('images/dtw_intragroup.png')
flush_matplotlib()

plt.hist(all_cos_known, 100)
plt.title('cosine distance between known groups')
plt.ylabel('cosine distance')
plt.savefig('images/cos_intragroup.png')
flush_matplotlib()













from scipy.spatial.distance import cosine

# Compute all dtw
to_seqs = lambda y: round((y*cqt_window)/(sr*timestep))
all_dtw = []
all_cos = []
for i, ag1 in tqdm.tqdm(list(enumerate(all_groups_over))):
    this_cos = []
    this_dtw = []
    for j, ag2 in enumerate(all_groups_over):
        if j >= i:
            continue
        sample1 = random.sample(ag1, min(10,len(ag1)))
        sample2 = random.sample(ag2, min(10,len(ag2)))
        av_dtw = []
        av_cos = []
        for (x0, x1), (y0, y1) in itertools.product(sample1, sample2):
            seq1 = np.trim_zeros(pitch[to_seqs(x0): to_seqs(x1)]) 
            seq2 = np.trim_zeros(pitch[to_seqs(y0): to_seqs(y1)])

            r1 = min([min(seq1), min(seq2)])
            r2 = max([max(seq2), max(seq2)])
            seq1_hist = np.histogram(seq1, 100, density=True, range=(r1, r2))[0]
            seq2_hist = np.histogram(seq2, 100, density=True, range=(r1, r2))[0]

            hist_dist = cosine(seq1_hist, seq2_hist)
            seq_len = min([len(seq1), len(seq2)])


            val,conn = fastdtw.fastdtw(seq1, seq2, radius=round(seq_len*0.5))
            av_dtw.append(val/len(conn))
            av_cos.append(hist_dist)

        this_dtw.append(np.mean(av_dtw))
        this_cos.append(np.mean(av_cos))
    all_dtw.append(this_dtw)
    all_cos.append(this_cos)


def get_dtw(thresh_dtw, thresh_cos):
    group_match_dict = {}
    for i, ag1 in tqdm.tqdm(list(enumerate(all_groups_over))):
        for j, ag2 in enumerate(all_groups_over):
            if j >= i:
                continue
            val_dtw = all_dtw[i][j]
            val_cos = all_cos[i][j]

            if val_dtw < thresh_dtw and val_cos > thresh_cos:
                update_dict(group_match_dict, i, j)
                update_dict(group_match_dict, j, i)

    for i in range(len(all_groups_over)):
        if i not in group_match_dict:
            group_match_dict[i] = []

    all_groups_ix = matches_dict_to_groups(group_match_dict)
    all_groups = [[x for i in group for x in all_groups_over[i]] for group in all_groups_ix]
    all_groups = [remove_group_duplicates(g, 0.75, True) for g in all_groups]

    return all_groups

# get plot for dtw
group_size = []
threshes = list(np.arange(0, 20, 0.01))
for thresh in threshes:
    group_match_dict = {}
    for i, ag1 in tqdm.tqdm(list(enumerate(all_groups_over))):
        for j, ag2 in enumerate(all_groups_over):
            if j >= i:
                continue
            val = that[i][j]

            if val < thresh:
                update_dict(group_match_dict, i, j)
                update_dict(group_match_dict, j, i)

    for i in range(len(all_groups_over)):
        if i not in group_match_dict:
            group_match_dict[i] = []

    all_groups_ix = matches_dict_to_groups(group_match_dict)
    all_groups = [[x for i in group for x in all_groups_over[i]] for group in all_groups_ix]
    all_groups = [remove_group_duplicates(g, 0.01) for g in all_groups]

    group_size.append(len(all_groups))

plt.plot(threshes, group_size)
plt.title('DTW Grouping')
plt.xlabel('DTW threshold')
plt.ylabel('Resultant number of groups')
plt.savefig('images/dtw_grouping.png')
flush_matplotlib()




x0 = starts_seq_ext[3][0]
x1 = x0 + lengths_seq_ext[3][0]

y0 = starts_seq_ext[1][1]
y1 = y0 + lengths_seq_ext[1][1]



seq1 = np.trim_zeros(pitch[x0: x1]) 
seq2 = np.trim_zeros(pitch[y0: y1])

r1 = min([min(seq1), min(seq2)])
r2 = max([max(seq2), max(seq2)])
seq1_hist,bins1 = np.histogram(seq1, 100, density=True, range=(max([r1,70]), min([r2,8000])))
seq2_hist,bins2 = np.histogram(seq2, 100, density=True, range=(max([r1,70]), min([r2,8000])))

hist_dist = cosine(seq1_hist, seq2_hist)
seq_len = min([len(seq1), len(seq2)])


val,conn = fastdtw.fastdtw(seq1, seq2, radius=round(seq_len*0.5))


plt.plot(bins1[1:],seq1_hist)
plt.title('Pitch histogram of matched sequence')
plt.ylabel('normalised count')
plt.xlabel('pitch (Hz)')
plt.savefig('images/seq1_match_hist.png')
flush_matplotlib()

plt.plot(bins2[1:],seq2_hist)
plt.title('Pitch histogram of matched sequence')
plt.ylabel('normalised count')
plt.xlabel('pitch (Hz)')
plt.savefig('images/seq2_match_hist.png')
flush_matplotlib()







#       Out[89]:
#       [(3, 78),
#        (4, 77),
#        (5, 76),
#        (6, 74),
#        (7, 68),
#        (8, 66),
#        (9, 64),
#        (10, 62),
#        (11, 60),
#        (12, 54),
#        (13, 47),
#        (14, 42),
#        (15, 33),
#        (16, 29),
#        (17, 23),
#        (18, 20),
#        (19, 15),
#        (20, 13),
#        (21, 11),
#        (22, 9),
#        (23, 7),
#        (24, 6),
#        (25, 5),
#        (26, 5),
#        (27, 4),
#        (28, 3),
#        (29, 3),
#        (30, 2),
#        (31, 2),
#        (32, 2),
#        (33, 2),
#        (34, 2),
#        (35, 2),
#        (36, 2),
#        (37, 2),
#        (38, 2),
#        (39, 2)]




thresh = 10

def runthresh(thresh, output=False):
    group_match_dict = {}
    for i, ag1 in tqdm.tqdm(list(enumerate(all_groups_over))):
        for j, ag2 in enumerate(all_groups_over):
            if j >= i:
                continue
            val = that[i][j]

            if val < thresh:
                update_dict(group_match_dict, i, j)
                update_dict(group_match_dict, j, i)

    for i in range(len(all_groups_over)):
        if i not in group_match_dict:
            group_match_dict[i] = []

    all_groups_ix = matches_dict_to_groups(group_match_dict)
    all_groups = [[x for i in group for x in all_groups_over[i]] for group in all_groups_ix]
    all_groups = [remove_group_duplicates(g, 0.01) for g in all_groups]
    print(f'Number of groups: {len(all_groups)}')

    all_groups_dtw = all_groups


    print('Convert sequences to pitch track timesteps')
    starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups_dtw, cqt_window, sr, timestep)

    print('Applying exclusion functions')
    #starts_seq_exc, lengths_seq_exc = apply_exclusions(raw_pitch, starts_seq, lengths_seq, exclusion_functions, min_in_group)
    starts_seq_exc,  lengths_seq_exc = remove_below_length(starts_seq, lengths_seq, timestep, min_pattern_length_seconds)

    starts_seq_exc = [p for p in starts_seq_exc if len(p)>min_in_group]
    lengths_seq_exc = [p for p in lengths_seq_exc if len(p)>min_in_group]

    print('Extend all segments to stable or silence')
    silence_and_stable_mask_2 = np.array([1 if any([i==2,j==2]) else 0 for i,j in zip(silence_mask, stable_mask)])
    starts_seq_ext, lengths_seq_ext = extend_to_mask(starts_seq_exc, lengths_seq_exc, silence_and_stable_mask_2)

    starts_sec_ext = [[x*timestep for x in p] for p in starts_seq_ext]
    lengths_sec_ext = [[x*timestep for x in l] for l in lengths_seq_ext]

    starts_seq_ext = [[int(x/timestep) for x in p] for p in starts_sec_ext]
    lengths_seq_ext = [[int(x/timestep) for x in l] for l in lengths_sec_ext]

    print('Evaluating')
    annotations_orig = load_annotations_new(annotations_path)
    if s1:
        annotations_filt = annotations_orig[(annotations_orig['s1']>=s1*cqt_window/sr) & (annotations_orig['s1']<=s2*cqt_window/sr)]
        annotations_filt['s1'] = annotations_filt['s1']-s1*cqt_window/sr
        annotations_filt['s2'] = annotations_filt['s2']-s1*cqt_window/sr
    else:
        annotations_filt = annotations_orig

    annotations_filt = annotations_filt[annotations_filt['tier']!='short_motif']
    #metrics, annotations_tag = evaluate_all_tiers(annotations_filt, starts_sec_exc, lengths_sec_exc, eval_tol, partial_perc)
    print('')
    n_patterns = sum([len(x) for x in starts_seq_ext])
    coverage = get_coverage(pitch, starts_seq_ext, lengths_seq_ext)
    print(f'Number of Patterns: {n_patterns}')
    print(f'Number of Groups: {len(starts_sec_ext)}')
    print(f'Coverage: {round(coverage,2)}')
    #evaluation_report(metrics)
    annotations_tagged, recall_all, precision_all, recall_motif, recall_phrase = evaluate_quick(annotations_filt, starts_sec_ext, lengths_sec_ext, eval_tol, partial_perc)

    if output:

        ############
        ## Output ##
        ############
        print('Writing all sequences')
        plot_all_sequences(raw_pitch, time, lengths_seq_ext[:top_n], starts_seq_ext[:top_n], 'output/new_group', clear_dir=True, plot_kwargs=plot_kwargs)
        #write_all_sequence_audio(audio_path, starts_seq_ext[:top_n], lengths_seq_ext[:top_n], timestep, 'output/new_group')
        annotations_tagged.to_csv('output/new_group/annotations.csv', index=False)
        flush_matplotlib()

    return len(starts_sec_ext), recall_all, precision_all, recall_motif, recall_phrase







all_recall = []
all_recall_phrase = []
all_recall_motif = []
all_precision = []
n_groups = []
all_thresh = np.arange(0,20, 0.01)
for thresh in tqdm.tqdm(all_thresh):
    n, recall_all, precision_all, recall_motif, recall_phrase = runthresh(thresh)
    all_recall.append(recall_all)
    all_precision.append(precision_all)
    all_recall_motif.append(recall_motif)
    all_recall_phrase.append(recall_phrase)
    n_groups.append(n)


plt.plot(all_thresh, all_recall, label = 'Recall')
plt.plot(all_thresh, all_recall_phrase, label = 'Recall (Phrase)')
plt.plot(all_thresh, all_recall_motif, label = 'Recall (Motif)')
plt.plot(all_thresh, all_precision, label = 'Precision')
#plt.plot(all_thresh, n_groups, label = 'Number of Groups')
plt.title('DTW grouping thresholds effect on performance')
plt.ylabel('DTW threshold for joining groups')
plt.xlabel('metric value')
plt.legend()
plt.savefig('images/DTW_thresh_evaluation.png')
flush_matplotlib()
