# Backscatter Tag Design for High Concurrency
## Task 1
After the hardware design, we receive multiple signals overlapped in the time domain. We extract the transition edges of different signals and then categorize these signals by their distinct frequencies.

![Figure_1](https://user-images.githubusercontent.com/107864216/222172083-6aeb05de-d1b9-4942-bf33-a5cd46ff3355.png)
### Generate Square Waveform
To simulate a square wave in real situations, ideal square is uncapable for its direct transitions in the edges. Therefore, a **smoothing** strategy, such as concoluted with a short window, should be taken:

```python
pwm = amp * signal.square(2 * np.pi * freq * t + phase)
window = signal.windows.hamming(somelen)
signal.convolve(pwm, window, mode='same')
```

Random noise can be added in window and final square waveform.

Besides, basic configuration including sampling frequency and simulation time, is saved as a dictionary.

### Extract edges
The received signal is always companied by channel noise. **Denoise** processing can be useful by decreasing the amplitude error of extracted edges ahead of **shift difference**. We implement this by convolution with hamming window.
