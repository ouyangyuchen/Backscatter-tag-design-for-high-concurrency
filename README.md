# Backscatter Tag Design for High Concurrency
Environment:
- python 3.10
- MATLAB R2022b (communication toolbox, signal processing toolbox)
## Get Sources
      > git clone https://github.com/ouyangyuchen/Backscatter-tag-design-for-high-concurrency.git
## Files
|Filename|description|
|---|---|
|*main.ipynb*|user can run the whole experiment with this script|
|*gen_wave.m*|generate and save waveform in MATLAB files|
|*extract.py*|extract edges from waveform and plot functions|
|*utils.py*|auxiliary functions, list below|
|*viterbi_decoding.py*|the implementation of classification algorithm and conditional probability function|
|*summary.pdf*|presentation slide for this project|

```
functions in utils.py

  - find_n_max        find the top n max entries in a matrix, used in the viterbi algorithm
  
  - find_pre_index    find the nearest entries with the same classes from end to start, used for averaging distances
  
  - plot_CDF          plot the distribution of classified edges
  
  - filtering         leach out the classes with too few edges and mark them to class 0
  
  - get_freq          calculate the frequencies of classified tags from averaged delta n.
  
  - freq_match        check the closest frequency in the ground truth for each classified tag.
  
  - count_acc         get measurements, like detected tags number and edges ratio.
```
