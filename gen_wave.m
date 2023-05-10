clear;
%% Configurations
% duration time, sampling freq, #tags
time = 0.002;
fs = 10000000;
num = 10;
t = linspace(0, time, time * fs);

SNR = 20;           % change this

SNR1 = 10 * log10(2 * num) + SNR;

% all config/signal save in dir:
if exist("signals/", 'dir') == 0
    mkdir("signals")
end

%% Freqs, Phases, Amplitudes
% 15kHz ~ 30kHz
freq = rand(1, num) * 15000 + 15000;
% amp = (rand(1, num) * .02 + .1) .* exp(1j * 2*pi * rand(1, num));
amp = [0.105278488140166 - 0.022815703894954i,0.116970245460998 - 0.010401511833459i,-0.051829899302457 + 0.093509992767321i,-0.098740847394605 + 0.022109307227178i,-0.057775834523463 - 0.081945277760733i,0.026574569729239 - 0.104211139979910i,0.062621920169006 + 0.080339654927814i,0.044419656911533 + 0.097900755864330i,0.106761404818690 + 0.003799455470436i,-0.102182347104108 - 0.059953844562520i];
phases = rand(1, num) * 2 * pi;
plot(real(amp), imag(amp), 'o');
hold on
plot(-real(amp), -imag(amp), 'o')
plot(0, 0, 'x');
hold off
%% Generate Wave
wave = zeros(1, length(t));
for i = 1:num
    pwm = amp(i) * square(2*pi * freq(i) * t + phases(i));
    pwm = awgn(pwm, SNR1, 'measured');
    window = 0.5 * hamming(3);
    pwm = conv(pwm, window, "same");
    wave = wave + pwm;
end
% plot(t(1:800), wave(1:800));
%hold off

path_square = sprintf("./signals/" + "tags%d_snr%d_db.mat", ...
    [num, SNR]);
save(path_square, "wave", "freq", "amp", "phases");
