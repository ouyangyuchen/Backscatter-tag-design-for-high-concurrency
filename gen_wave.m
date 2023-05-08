clear;
%% Configurations
% duration time, sampling freq, #tags
time = 0.0002;
fs = 10000000;
num = 20;
t = linspace(0, time, time * fs);

winlen = 3;
noise = 0.5;
edge_noise = 0.2;

SNR = 40;

% all config/signal save in dir:
if exist("signals/", 'dir') == 0
    mkdir("signals")
end

%% Freqs, Phases, Amplitudes
% 15kHz ~ 30kHz
freq = (1:num) * 1500 + 30000;
amp = (rand(1, num) * 5 + 5) .* exp(1j * 2*pi * rand(1, num));
phases = rand(1, num) * 2 * pi;
% plot(real(amp), imag(amp), '.');

%% Generate Wave
wave = zeros(1, length(t));
for i = 1:num
    pwm = amp(i) * square(2*pi * freq(i) * t + phases(i));
    window = hamming(winlen) + edge_noise * randn(winlen, 1);
    pwm = conv(pwm, window, "same");
    wave = wave + pwm;
end
% gaussian noise to dB
wave = awgn(wave, SNR, 'measured');
% plot(t(1:800), wave(1:800));
%hold off

path_square = sprintf("./signals/" + "tags%d_snr%d_db.mat", ...
    [num, SNR]);
save(path_square, "wave", "freq", "amp", "phases");
