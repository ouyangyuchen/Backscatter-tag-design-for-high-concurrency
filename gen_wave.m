%% Configurations
% duration time, sampling freq, #tags
time = 0.00004;
fs = 10000000;
num = 20;
t = linspace(0, time, time * fs);

winlen = 3;
noise = 0.5;
edge_noise = 0.2;

% all config/signal save in dir:
if exist("signals/", 'dir') == 0
    mkdir("signals")
end

%% Freqs, Phases, Amplitudes
% 15kHz ~ 30kHz
freq = (1:num) * 15000 + 30000;
amp = (rand(1, num) * 5 + 5) .* exp(1j * 2*pi * rand(1, num));
phases = rand(1, num) * 2 * pi;
%plot(real(amp), imag(amp), '.');

save("signals/config.mat", "freq", "amp", "phases");

%% Generate Wave
wave = zeros(1, length(t));
for i = 1:num
    pwm = amp(i) * square(2*pi * freq(i) * t + phases(i));
    window = hamming(winlen) + edge_noise * randn(winlen, 1);
    pwm = conv(pwm, window, "same");
    wave = wave + pwm;
end
% gaussian noise
wave = wave + noise * randn(1, length(t));
%plot(t(1:400), wave(1:400), '-o');
%hold off

path_square = sprintf("signals/tags%d_noise_%3.2f_%d.mat", [num, noise, length(t)]);
save(path_square, "wave");
