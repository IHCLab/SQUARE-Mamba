function curve_plot(gt, prediction, index, mode)
    color = [255/255, 105/255, 0/255];
    years = ["2007-12", "2008-12", "2009-12", "2010-12", "2011-12", "2012-12", "2013-12", "2014-12", "2015-12", "2016-12", "2017-12", "2018-12", "2019-12", "2020-12", "2021-12", "2022-12", "2023-12"];
    figure;
    set(gcf, 'Position', [100, 100, 1870, 400]); movegui(gcf, 'center'); hold on;
    plot(gt, 'Color', [0, 0, 0], 'LineWidth', 1.5);
    plot(prediction, 'Color', color, 'LineStyle', '--', 'LineWidth', 2.3, 'Marker', 'o', 'MarkerSize', 4, 'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'MarkerIndices', 1:1:length(prediction));
    %
    title_str = strrep(mode, '_', '\_');
    lgd = legend({'Observed', title_str}, 'FontSize', 11, 'FontName', 'Times New Roman');
    set(lgd, 'Position', [0.15, 0.8, 0.1, 0.1]);
    set(gca, 'YLim', [-2.3, 3], 'FontName', 'Times New Roman', 'FontSize', 14, 'GridColor', [0, 0, 0], 'GridAlpha', 0.3, 'LineWidth', 1);
    xticks(1:12:length(gt));  xticklabels(years);
    ylabel('Drought Forecasting', 'FontName', 'Times New Roman', 'FontSize', 14);
    title(title_str, 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 18)
    grid on; box on; axis([1 numel(gt) -3 3]);
    %
    text_x = 195; text_y = 2.5;  
    text(text_x, text_y, sprintf('MAE = %.4f', index.MAE), 'FontSize', 16, 'FontName', 'Times New Roman');
    text(text_x, text_y-2.5 , sprintf('RMSE = %.4f', index.RMSE), 'FontSize', 16, 'FontName', 'Times New Roman');
    text(text_x, text_y-5 , sprintf('R^2 = %.4f', index.R2), 'FontSize', 16, 'FontName', 'Times New Roman');
end