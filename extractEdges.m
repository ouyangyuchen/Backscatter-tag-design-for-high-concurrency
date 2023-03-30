%% Load square wave
time = 0.0004;

clear wave
load("signals/tags20_noise_0.50_400.mat", "wave");

t = linspace(0, time, length(wave));
plot(t, wave);
hold off

%% Extract Edges
shift = 1;
temp = circshift(wave, shift, 2);
temp(1:shift) = 0;
impulses = wave - temp;               % shift and subtract
impulses = complex(abs(real(impulses)), abs(imag(impulses)));

plot(t, real(impulses), '-.');
hold on
plot(t, imag(impulses), '-o');
hold off