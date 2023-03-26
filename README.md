## In Search of Sañcāras - Tradition-Informed Repeated Melodic Pattern Recognition in Carnatic Music

This repository contains the accompanying code for the ISMIR 2022 submission:

`A. Anonymous: In Search of Sañcāras - Tradition-Informed Repeated Melodic Pattern Recognition in Carnatic Music. [In Review for the] Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022, Bangalore, India.`

### 1. Results Explorer
You can explore all results presented in the paper, and more, using the Google Colab notebook [here](https://colab.research.google.com/drive/115wznvNTr0cdaKN3EBWuCJMz3n-A7P-J?usp=sharing). This includes pitch plots and audio corresponding to the pattern groups returned for a selection of Carnatic performances in the [Saraga Dataset](https://mtg.github.io/saraga/).

The pitch plots corresponding to the paper results are also available in `ouput/for_paper/pitch_plots/`.

### 2. Data
The datasets and models presented in the paper are available at the following...

| **Data**                     | **Description**                                                     | **Location**                                |
|------------------------------|---------------------------------------------------------------------|---------------------------------------------|
| Annotations                  | Expert annotations of 3 Carnatic performances in SCV                | [`annotations/`]https://github.com/MTG/searching_for_sancaras/tree/main/annotations)              |
| Saraga Carnatic Melody Synth | SCMS dataset of synthesized predominant pitch ground-truth          | [zenodo](https://zenodo.org/record/5553925) |
| Saraga Carnatic Vocal        | SCV dataset of performances for which we have multitrack recordings | [link](url)								   |
| CAE Model                    | Complex Autoencoder (CAE) trained on SCV                            | [link](url)                                 |
| Pitch Tracks                 | Predominant pitch tracks extracted using FTA-NET trained on SCMS    | [`data/pitch_tracks`](https://github.com/searching-sancaras-ISMIR22/searching_for_sancaras/tree/main/data/pitch_tracks)                         |
| Silence/Stability Masks      | Mask annotated silent or stable regions in pitch tracks             | [`data/stability_tracks`](https://github.com/searching-sancaras-ISMIR22/searching_for_sancaras/tree/main/data/stability_tracks)              |
| CAE Features                 | Features for all of Saraga extracted using CAE model trained on SCV | [zenodo](url)                               |
| Source Separated Audio       | SCV dataset after Spleeter source separation                        | [zenodo](url)                               |

**Table 1** - Data relevant to task

### 3. Pipeline Overview

![Overview of pipeline](./plots_for_paper/schematic.png?raw=true)

**Figure 1** - Overview of pipeline


### 4. Code Usage

#### 4.1 Install
Requires Python 3.8, to install requirements...

`pip install -r requirements.txt`

#### 4.2 Relevant dependencies
The pipeline requires a model trained using the Complex Autoencoder architecture [[github]](https://github.com/SonyCSLParis/cae-invar) and pitch annotations of the predominant melodic line extracted using FTA-NET trained on the SCMS dataset [[github]](https://github.com/TISMIR22-Carnatic/carnatic-pitch-patterns)  (see **Table 1** in **2. Data** for data and models)

#### 4.3 Configure

To configure the pipeline, update the configuration files in `conf/`. In that folder you will find more details on each parameters function.

#### 4.4 Run

##### 4.4.1 Feature Extraction
To extract the silence/stability mask for a pitch track, `<folder>/<pitch_track>.csv`, using the parameters specified in `conf/mask.yaml`:

```
python src mask '<folder>/<pitch_track>.csv' --config 'conf/mask.yaml'
```

To extract the self similarity matrix for an audio, `<folder>/<audio>.mp3`, masked with `<folder>/<mask>.csv`, using the parameters specified in `conf/selfsim.yaml`:

```
python src selfsim '<folder>/<audio>.mp3' '<folder>/<mask>.csv' --config 'conf/selfsim.yaml'
```

##### 4.4.1 Pattern Discovery

To run the melodic pattern extraction pipeline using the parameters specified in `conf/pattern.yaml`:

```
python src pattern --config conf/pattern.yaml
```

There are more detailed explanations of each parameter in the `conf/` directory.

##### 4.4.2 Output

Results will be saved to the output directory specified in `conf/pattern.yaml`. This directory will contain folder groups of repeated melodic patterns, each folder contains the audios of each pattern and the corresponding pitch plot (Figure 2).

![Occurrence 1, motif group 2, Sharanu Janakana, pdmmmmgrrgdpmgrssnd](output/for_paper/pitch_plots/Sharanu%20Janakana/motif_2_len%3D9.5/1_time%3D4min-24.89sec.png?raw=true)

**Figure 2** - Sharanu Janakana. Occurrence 1 in motif group 2 corresponding to the characteristic phrase, pdmmmmgrrgdpmgrssnd.

The y-axis represents cents above the tonic (S), divided into the discrete pitch positions defined in Carnatic music theory for the corresponding rāga (in this case, Bilahari). The grey region of the pitch plot corresponds to the returned pattern, the surrounding areas are included to present the pattern within it's melodic context and are not considered part of the returned pattern nor are they included in the associated audio.

Also included in the output directory is an array of start points and lengths for each pattern in each group and - if annotations are passed - the annotations dataframe with a column indicating whether the corresponding pattern was matched by the process and if so, its group and occurrence number.

All outputs can be switched on/off in `conf/pattern.yaml`

### 5. Reproducibility

`experiments/` contains various one-off scripts related to the development process and paper. The following are relevant to reproducing the results in the paper...	

Extract silence/stability masks - `python experiments/mask_extraction.py`

Extract CAE features - `python experiments/cae_features.py`

Gridsearch for optimum parameters - `python experiments/gridsearch.py`

Generate results for performances in paper - `python experiments/get_results.py`

Reproduce plots in paper - `python experiments/plots_for_paper.py`

Generate results for various un-annotated tracks - `python experiments/random_results.py`

Investigate silent and stable regions in annotations - `python experiments/stability_and_silence_annotations.py`

Evaluate - `python experiments/evaluate.py`

### 6. References

[1] Yu, S., Sun, X., Yu, Y., and Li, W. (May, 2021). Frequency-Temporal Attention Network for Singing Melody Extraction. Paper presented at the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Toronto, Canada.

[2] S. Lattner, A. Arzt, and M. Dörfler, “Learning complex basis functions for invariant representations of audio,” Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR), Delft, The Netherlands, 2019.
