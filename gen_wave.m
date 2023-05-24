clear;
%% Configurations
% duration time, sampling freq, #tags
time = 0.002;
fs = 10000000;
num = 15;
t = linspace(0, time, time * fs);

SNR = 15;           % change this

SNR1 = 10 * log10(2 * num) + SNR;

% all config/signal save in dir:
if exist("signals/", 'dir') == 0
    mkdir("signals")
end

%% Freqs, Phases, Amplitudes
% 15kHz ~ 30kHz
freq = (0:num-1) * 15000 / num + 15000 + rand(1, num) * 200 - 100;

% amp = (rand(1, num) * .15 + .05) .* exp(1j * 2*pi * rand(1, num));

% tags=15, 1
amp = [-0.219376017327974 + 0.144691683662132i,0.00151912539268548 + 0.110023730785698i,-0.183674239821316 - 0.203284771536662i,-0.117424691912982 + 0.20606575624250i,-0.163516939000385 - 0.0403610744421877i,0.185011568733075 + 0.120571253353260i,-0.0989019988009539 + 0.0714371270675762i,-0.277351730382089 + 0.0860708371443860i,-0.113305457069267 - 0.252575059174384i,-0.106043119392994 + 0.133382142038544i,0.0358256365913430 + 0.177423970968777i,-0.0246995971564955 + 0.208224302551439i,-0.190288237154665 + 0.023582350327914i,...
    -0.111384723292813 - 0.120642109167877i,0.0609158163257177 + 0.240632731104878i];

% tags=10, bad constellation
% amp = [0.105278488140166 - 0.022815703894954i,0.116970245460998 - 0.010401511833459i,-0.051829899302457 + 0.093509992767321i,-0.098740847394605 + 0.022109307227178i,-0.057775834523463 - 0.081945277760733i,0.026574569729239 - 0.104211139979910i,0.062621920169006 + 0.080339654927814i,0.044419656911533 + 0.097900755864330i,0.106761404818690 + 0.003799455470436i,-0.102182347104108 - 0.059953844562520i];

phases = rand(1, num) * 2 * pi;
plot(real(amp), imag(amp), 'o');
hold on
plot(-real(amp), -imag(amp), 'o')
plot(0, 0, 'x');
title("Amplitudes in I-Q domain")
text(-0.25, 0.27, "avr\_freq=22.5kHz")
xlabel("Real")
ylabel("Imag")
hold off
%% Generate Wave
wave = zeros(1, length(t));
for i = 1:num
    pwm = amp(i) * square(2*pi * freq(i) * t + phases(i));
    pwm = awgn(pwm, SNR1, 'measured');
    window = hamming(3);
    pwm = conv(pwm, window, "same");
    wave = wave + pwm;
end
% plot(t(1:800), wave(1:800));
%hold off

path_square = sprintf("./signals/" + "tags%d_snr%d_db.mat", ...
    [num, SNR]);
save(path_square, "wave", "freq", "amp", "phases");
