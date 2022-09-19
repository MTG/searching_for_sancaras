import os
import sys

# non-relative path to cae-invar directory with model trained
path_to_cae = '/Volumes/Shruti/asplab2/original_cae/cae-invar/'

# keyword of model trained
run_keyword = 'hpc'
sys.path.append(f'{path_to_cae}')

import pandas as pd

from extract_motives import process_audio_poly
from complex_auto.util import read_file

from experiments.segraw_to_segments import segraw_to_segments
from experiments.baseline_comparison import run_pattern

rho = [2]
tol = [0.01, 0.05, 0.1, 0.5, 1.0, 1.1, 1.5]

results = pd.DataFrame()

for r in rho:
	for t in tol:
		print('\n\n\n\n\n')
		print(f'r: {r}, t: {t}')
		out_dir = os.path.join(f"{path_to_cae}/output", run_keyword)
		input_filelist = os.path.join(out_dir, "ss_matrices_filelist_koti.txt")
		inputs = read_file(input_filelist)
		inputs = [os.path.join(f'{path_to_cae}', i) for i in inputs]

		process_audio_poly(inputs, out_dir, tol=t, rho=r,
		                       domain='audio', csv_files=None,
		                       ssm_read_pk=False, read_pk=False,
		                       n_jobs=10, tonnetz=False)

		seg_path = inputs[0].replace('mp3.npy','segraw')

		segraw_to_segments(seg_path)
		recall, precision, f1, _ = run_pattern(1)
		results = results.append({
				'recall': recall,
				'precision': precision, 
				'f1': f1,
				'rho':r,
				'tol':t
			}, ignore_index=True)

results.to_csv('results_baseline.csv', index=False)