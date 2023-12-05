figure;
subplot(3,1,1);
conf = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
fix_conf_idf1 = [76.5 78.1 80.0 80.4 81.6 81.9 71.1];
fix_conf_mota = [83.3 84.6 85.7 86.3 86.4 84.0 64.3];
fix_conf_ids = [826 731 645 564 425 294 106];
plot(conf, fix_conf_idf1, 'r-*'); hold on;
plot(conf, ones(size(conf)) * 81.3, 'r-');
plot(conf, fix_conf_mota, 'b-.*'); hold on;
plot(conf, ones(size(conf)) * 86.3, 'b-.');
xlim([min(conf), max(conf)]);
xlabel('Confidence Score');
ylabel('Performance in Percentage');
legend('Fixed Conf. IDF1', 'Adaptive Conf. IDF1', 'Fixed Conf. MOTA', 'Adaptive Conf. MOTA');
title('Evaluation Comparion on MOT16 Dataset');
hold on;
subplot(3,1,2);
conf = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
fix_conf_idf1 = [76.7 78.4 80.2 80.6 81.9 81.7 70.5];
fix_conf_mota = [84.8 86.0 87.1 87.6 87.4 84.2 63.6];
fix_conf_ids = [2613 2292 2007 1740 1308 909 324];
plot(conf, fix_conf_idf1, 'r-*'); hold on;
plot(conf, ones(size(conf)) * 81.5, 'r-');
plot(conf, fix_conf_mota, 'b-.*'); hold on;
plot(conf, ones(size(conf)) * 87.5, 'b-.');
xlim([min(conf), max(conf)]);
xlabel('Confidence Score');
ylabel('Performance in Percentage');
legend('Fixed Conf. IDF1', 'Adaptive Conf. IDF1', 'Fixed Conf. MOTA', 'Adaptive Conf. MOTA');
title('Evaluation Comparion on MOT17 Dataset');
hold on;
subplot(3,1,3);
conf = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
fix_conf_idf1 = [80.9 81.2 81.7 81.9 81.9 81.7 60.8];
fix_conf_mota = [78.5 79.0 79.2 79.4 78.9 78.9 53.0];
fix_conf_ids = [2003 1599 1291 1129 1050 934 560];
plot(conf, fix_conf_idf1, 'r-*'); hold on;
plot(conf, ones(size(conf)) * 81.6, 'r-');
plot(conf, fix_conf_mota, 'b-.*'); hold on;
plot(conf, ones(size(conf)) * 79.4, 'b-.');
xlim([min(conf), max(conf)]);
xlabel('Confidence Score');
ylabel('Performance in Percentage');
legend('Fixed Conf. IDF1', 'Adaptive Conf. IDF1', 'Fixed Conf. MOTA', 'Adaptive Conf. MOTA');
title('Evaluation Comparion on MOT20 Dataset');

set(gcf, 'Color', 'w');
addpath(genpath('./export_fig'));
%export_fig conf_selection.pdf; % https://github.com/altmany/export_fig
