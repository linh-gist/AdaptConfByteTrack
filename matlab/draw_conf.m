headerlinesIn = 0;
delimiterIn = '';

MOT16_04_f50 = importdata("MOT16-04_f50.txt", delimiterIn, headerlinesIn);
MOT16_13_f250 = importdata("MOT16-13_f250.txt", delimiterIn, headerlinesIn);
MOT20_01_f100 = importdata("MOT20-01_f100.txt", delimiterIn, headerlinesIn);

figure;
subplot(3,1,1);
plot(1:size(MOT16_04_f50,1),MOT16_04_f50)
title('MOT16-04, Frame 50');
xlabel('Detection');
ylabel('Conf. Score');
[~, argmin] = min(diff(MOT16_04_f50));
hold on;
plot(argmin, MOT16_04_f50(argmin), 'ro');

hold on;
subplot(3,1,2); 
plot(1:size(MOT16_13_f250,1),MOT16_13_f250)
title('MOT16-13, Frame 250');
xlabel('Detection');
ylabel('Conf. Score');
[~, argmin] = min(diff(MOT16_13_f250));
hold on;
plot(argmin, MOT16_13_f250(argmin), 'ro');

hold on;
subplot(3,1,3); 
plot(1:size(MOT20_01_f100,1),MOT20_01_f100)
title('MOT20-01, Frame 100');
xlabel('Detection');
ylabel('Conf. Score');
[~, argmin] = min(diff(MOT20_01_f100));
hold on;
plot(argmin, MOT20_01_f100(argmin), 'ro');

set(gcf, 'Color', 'w');
addpath(genpath('./export_fig'));
%export_fig conf_selection.pdf; % https://github.com/altmany/export_fig

