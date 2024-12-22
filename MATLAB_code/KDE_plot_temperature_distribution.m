load("monthly_mean_till_2023.mat")

Tmax_new = matrix;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check the Normality of the Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original

% Extract years
years = Tmax_new(:,1);
unique_years = unique(years);
num_years = length(unique_years);

% Initialize a figure
figure;
hold on;

% Set color map for visual effect
colors = jet(num_years);  % Each year gets a distinct color

% Pre-allocate arrays for efficiency
xi_arrays = cell(num_years, 1);
f_scaled_arrays = cell(num_years, 1);

% Determine global min and max temperature range for consistent x-axis across plots
all_temps = Tmax_new(:, 3:end);
all_temps = all_temps(:);  % Flatten the matrix to a vector
all_temps = all_temps(~isnan(all_temps));  % Remove NaN values
[f_all, xi_all] = ksdensity(all_temps, 'Bandwidth', 0.5);  % Using global data to determine range
x_range = [min(xi_all) max(xi_all)];  % Determining the x-range for consistent plotting

% Loop through each year to calculate KDE data first
for i = 1:num_years
    year = unique_years(i);
    % Extract data for the current year
    idx = years == year;
    daily_temps = Tmax_new(idx, 3:end);
    daily_temps = daily_temps(:);  % Flatten the matrix to a vector
    daily_temps = daily_temps(~isnan(daily_temps));  % Remove NaN values

    % Compute Kernel Density Estimation (KDE)
    [f, xi] = ksdensity(daily_temps, 'Bandwidth', 0.5, 'Support', 'unbounded');  % Unbounded support

    % Normalize the KDE values for better visibility in the plot
    f_scaled = f / max(f) * 5;  % Normalize and scale

    % Store the computed values for later use
    xi_arrays{i} = xi;
    f_scaled_arrays{i} = f_scaled + i;
    
    % Fill under the KDE curve
    fill(xi, f_scaled + i, colors(i,:), 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end

% After all fills, plot the lines to ensure they are on top
for i = 1:num_years
    plot(xi_arrays{i}, f_scaled_arrays{i}, 'k', 'LineWidth', 1.5);  % Black line for the KDE
end

% Customize the plot
set(gca, 'YTick', 1:5:num_years, 'YTickLabel', arrayfun(@(y) num2str(y, '%d'), unique_years(1:5:end), 'UniformOutput', false), 'FontSize', 10);
xlabel('Temperature (°C)', 'FontSize', 12);
ylabel('Year', 'FontSize', 12);
title('a)', 'FontSize', 18);
xlim(x_range);  % Set x-limits to show the full range of data
ylim([1, num_years + 5]);  % Extend the y-axis to provide space above 2017
grid on; % Turn on the grid

hold off;

% Adjust figure size and orientation
set(gcf, 'Position', [100, 100, 600, 2000]); % Set figure position and size in pixels
print(gcf, 'Annual Distribution of Daily Maximum Temperatures (1984-2023).jpg', '-djpeg', '-r300');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modified graph
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract years
years = Tmax_new(:,1);
unique_years = unique(years);
num_years = length(unique_years);

% Initialize a figure
figure;
hold on;

% Pre-allocate arrays for efficiency
xi_arrays = cell(num_years, 1);
f_scaled_arrays = cell(num_years, 1);

% Determine global min and max temperature range for consistent x-axis across plots
all_temps = Tmax_new(:, 3:end);
all_temps = all_temps(:);  % Flatten the matrix to a vector
all_temps = all_temps(~isnan(all_temps));  % Remove NaN values
[f_all, xi_all] = ksdensity(all_temps, 'Bandwidth', 0.5);  % Using global data to determine range
x_range = [min(xi_all) max(xi_all)];  % Determining the x-range for consistent plotting

% Define the colormap: transition from blue (low) to yellow (medium) to red (high)
cmap = jet(256);  % Jet colormap with 256 colors

% Define the threshold temperature for color transitions
threshold_temp = 34;  % Threshold temperature for color transition
cmap_low = cmap(1:round(256 * 0.95), :);  % Blue to yellow transition
cmap_high = cmap(round(256 * 0.95):end, :);  % Yellow to red transition

% Loop through each year to calculate KDE data first
for i = 1:num_years
    year = unique_years(i);
    % Extract data for the current year
    idx = years == year;
    daily_temps = Tmax_new(idx, 3:end);
    daily_temps = daily_temps(:);  % Flatten the matrix to a vector
    daily_temps = daily_temps(~isnan(daily_temps));  % Remove NaN values

    % Compute Kernel Density Estimation (KDE)
    [f, xi] = ksdensity(daily_temps, 'Bandwidth', 0.5, 'Support', 'unbounded');  % Unbounded support

    % Normalize the KDE values for better visibility in the plot
    f_scaled = f / max(f) * 5;  % Normalize and scale

    % Plot the KDE curve in segments with different colors
    for j = 1:length(xi)-1
        % Determine the color for the segment based on temperature value
        if xi(j) <= threshold_temp
            temp_norm = (xi(j) - x_range(1)) / (threshold_temp - x_range(1));  % Normalize xi to a 0-1 range for low temperatures
            color_idx = round(temp_norm * (size(cmap_low, 1) - 1)) + 1;  % Map normalized xi to colormap indices for low temperatures
            color = cmap_low(color_idx, :);  % Get the color from the low-temperature colormap
        else
            temp_norm = (xi(j) - threshold_temp) / (x_range(2) - threshold_temp);  % Normalize xi to a 0-1 range for high temperatures
            color_idx = round(temp_norm * (size(cmap_high, 1) - 1)) + 1;  % Map normalized xi to colormap indices for high temperatures
            color = cmap_high(color_idx, :);  % Get the color from the high-temperature colormap
        end
        
        % Plot the segment of the KDE
        fill([xi(j) xi(j+1) xi(j+1) xi(j)], [f_scaled(j)+i f_scaled(j+1)+i i i], color, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end

    % Store the computed values for later use
    xi_arrays{i} = xi;
    f_scaled_arrays{i} = f_scaled + i;
end

% After all fills, plot the lines to ensure they are on top
for i = 1:num_years
    plot(xi_arrays{i}, f_scaled_arrays{i}, 'k', 'LineWidth', 1.5);  % Black line for the KDE
end

% Customize the plot
set(gca, 'YTick', 1:5:num_years, 'YTickLabel', arrayfun(@(y) num2str(y, '%d'), unique_years(1:5:end), 'UniformOutput', false), 'FontSize', 10);
xlabel('Temperature (°C)', 'FontSize', 12);
ylabel('Year', 'FontSize', 12);
title('a)', 'FontSize', 18);
xlim(x_range);  % Set x-limits to show the full range of data
ylim([1, num_years + 5]);  % Extend the y-axis to provide space above 2017
grid on; % Turn on the grid

hold off;

% Adjust figure size and orientation
set(gcf, 'Position', [100, 100, 600, 2000]); % Set figure position and size in pixels
print(gcf, 'Annual Distribution of Daily Maximum Temperatures (1984-2023).jpg', '-djpeg', '-r300');