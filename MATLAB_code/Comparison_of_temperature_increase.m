%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ERE621 - Spatial Analysis
% Taposh Mollick
% Class project
% Ordinary Kriging for Temperature prediction using Bangladesh BMD Data
% Comparison of temperature increse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

load("Bangladesh_BMD_Tmax_35_years.mat")
load("Mean_temp_April_January_1984-2017.mat");

Matp;
boundaryMat = BGD_division_boundary;
stations = BMD_Stations_Elevation;

b_a_summer = (summer1996prtemp(:,3))-(summer1984prtemp(:,3));
c_b_summer = (summer2008prtemp(:,3))-(summer1996prtemp(:,3));
c_a_summer = (summer2008prtemp(:,3))-(summer1984prtemp(:,3));


b_a_winter = (winter1996prtemp(:,3))-(winter1984prtemp(:,3));
c_b_winter = (winter2008prtemp(:,3))-(winter1996prtemp(:,3));
c_a_winter = (winter2008prtemp(:,3))-(winter1984prtemp(:,3));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% b-a summer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Longitude and Latitude from Matp
longitudes = summer1984prtemp(:, 1);
latitudes = summer1984prtemp(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, b_a_summer, 'linear', 'linear');
temp_grid = F(long_grid, lat_grid);

% Mask out points outside the boundary using inpolygon
inBoundary = inpolygon(long_grid, lat_grid, boundaryMat(:,1), boundaryMat(:,2));
temp_grid(~inBoundary) = NaN; % Assign NaN for points outside the boundary

% Define contour levels
min_temp = min(temp_grid(:));
max_temp = max(temp_grid(:));
contour_levels = linspace(min_temp, max_temp, 70); % Adjust the number of levels for smoothness

% Plot the filled contour map
figure;
contourf(long_grid, lat_grid, temp_grid, contour_levels, 'LineStyle', 'none');
colormap(jet); % Apply the jet colormap
colorbar; % Show colorbar
caxis([min_temp, max_temp]); % Set the color axis scaling to match the temperature range

% Overlay the boundary
hold on;
scatter(stations(:, 1), stations(:, 2), 'magenta', 'filled');
plot(boundaryMat(:,1), boundaryMat(:,2), 'k-', 'LineWidth', 0.2); % Plot the boundary in black

% Add labels and title to the plot
xlabel('Longitude');
ylabel('Latitude');
title('d) April (b-a)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, 'April (b-a).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c-b summer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Longitude and Latitude from Matp
longitudes = summer1996prtemp(:, 1);
latitudes = summer1996prtemp(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, c_b_summer, 'linear', 'linear');
temp_grid = F(long_grid, lat_grid);

% Mask out points outside the boundary using inpolygon
inBoundary = inpolygon(long_grid, lat_grid, boundaryMat(:,1), boundaryMat(:,2));
temp_grid(~inBoundary) = NaN; % Assign NaN for points outside the boundary

% Define contour levels
min_temp = min(temp_grid(:));
max_temp = max(temp_grid(:));
contour_levels = linspace(min_temp, max_temp, 70); % Adjust the number of levels for smoothness

% Plot the filled contour map
figure;
contourf(long_grid, lat_grid, temp_grid, contour_levels, 'LineStyle', 'none');
colormap(jet); % Apply the jet colormap
colorbar; % Show colorbar
caxis([min_temp, max_temp]); % Set the color axis scaling to match the temperature range

% Overlay the boundary
hold on;
scatter(stations(:, 1), stations(:, 2), 'magenta', 'filled');
plot(boundaryMat(:,1), boundaryMat(:,2), 'k-', 'LineWidth', 0.2); % Plot the boundary in black

% Add labels and title to the plot
xlabel('Longitude');
ylabel('Latitude');
title('e) April (c - b)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, 'April (c - b).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c-a summer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Longitude and Latitude from Matp
longitudes = summer2008prtemp(:, 1);
latitudes = summer2008prtemp(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, c_a_summer, 'linear', 'linear');
temp_grid = F(long_grid, lat_grid);

% Mask out points outside the boundary using inpolygon
inBoundary = inpolygon(long_grid, lat_grid, boundaryMat(:,1), boundaryMat(:,2));
temp_grid(~inBoundary) = NaN; % Assign NaN for points outside the boundary

% Define contour levels
min_temp = min(temp_grid(:));
max_temp = max(temp_grid(:));
contour_levels = linspace(min_temp, max_temp, 70); % Adjust the number of levels for smoothness

% Plot the filled contour map
figure;
contourf(long_grid, lat_grid, temp_grid, contour_levels, 'LineStyle', 'none');
colormap(jet); % Apply the jet colormap
colorbar; % Show colorbar
caxis([min_temp, max_temp]); % Set the color axis scaling to match the temperature range

% Overlay the boundary
hold on;
scatter(stations(:, 1), stations(:, 2), 'magenta', 'filled');
plot(boundaryMat(:,1), boundaryMat(:,2), 'k-', 'LineWidth', 0.2); % Plot the boundary in black

% Add labels and title to the plot
xlabel('Longitude');
ylabel('Latitude');
title('f) April (c - a)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, 'April (c - a).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% b-a winter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Longitude and Latitude from Matp
longitudes = winter1984prtemp(:, 1);
latitudes = winter1984prtemp(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, b_a_winter, 'linear', 'linear');
temp_grid = F(long_grid, lat_grid);

% Mask out points outside the boundary using inpolygon
inBoundary = inpolygon(long_grid, lat_grid, boundaryMat(:,1), boundaryMat(:,2));
temp_grid(~inBoundary) = NaN; % Assign NaN for points outside the boundary

% Define contour levels
min_temp = min(temp_grid(:));
max_temp = max(temp_grid(:));
contour_levels = linspace(min_temp, max_temp, 70); % Adjust the number of levels for smoothness

% Plot the filled contour map
figure;
contourf(long_grid, lat_grid, temp_grid, contour_levels, 'LineStyle', 'none');
colormap(jet); % Apply the jet colormap
colorbar; % Show colorbar
caxis([min_temp, max_temp]); % Set the color axis scaling to match the temperature range

% Overlay the boundary
hold on;
scatter(stations(:, 1), stations(:, 2), 'magenta', 'filled');
plot(boundaryMat(:,1), boundaryMat(:,2), 'k-', 'LineWidth', 0.2); % Plot the boundary in black

% Add labels and title to the plot
xlabel('Longitude');
ylabel('Latitude');
title('d) January (b - a)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, 'January (b - a).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c-b winter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Longitude and Latitude from Matp
longitudes = winter1996prtemp(:, 1);
latitudes = winter1996prtemp(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, c_b_winter, 'linear', 'linear');
temp_grid = F(long_grid, lat_grid);

% Mask out points outside the boundary using inpolygon
inBoundary = inpolygon(long_grid, lat_grid, boundaryMat(:,1), boundaryMat(:,2));
temp_grid(~inBoundary) = NaN; % Assign NaN for points outside the boundary

% Define contour levels
min_temp = min(temp_grid(:));
max_temp = max(temp_grid(:));
contour_levels = linspace(min_temp, max_temp, 70); % Adjust the number of levels for smoothness

% Plot the filled contour map
figure;
contourf(long_grid, lat_grid, temp_grid, contour_levels, 'LineStyle', 'none');
colormap(jet); % Apply the jet colormap
colorbar; % Show colorbar
caxis([min_temp, max_temp]); % Set the color axis scaling to match the temperature range

% Overlay the boundary
hold on;
scatter(stations(:, 1), stations(:, 2), 'magenta', 'filled');
plot(boundaryMat(:,1), boundaryMat(:,2), 'k-', 'LineWidth', 0.2); % Plot the boundary in black

% Add labels and title to the plot
xlabel('Longitude');
ylabel('Latitude');
title('e) January (c - b)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, 'January (c - b).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c-a winter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Longitude and Latitude from Matp
longitudes = winter2008prtemp(:, 1);
latitudes = winter2008prtemp(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, c_a_winter, 'linear', 'linear');
temp_grid = F(long_grid, lat_grid);

% Mask out points outside the boundary using inpolygon
inBoundary = inpolygon(long_grid, lat_grid, boundaryMat(:,1), boundaryMat(:,2));
temp_grid(~inBoundary) = NaN; % Assign NaN for points outside the boundary

% Define contour levels
min_temp = min(temp_grid(:));
max_temp = max(temp_grid(:));
contour_levels = linspace(min_temp, max_temp, 70); % Adjust the number of levels for smoothness

% Plot the filled contour map
figure;
contourf(long_grid, lat_grid, temp_grid, contour_levels, 'LineStyle', 'none');
colormap(jet); % Apply the jet colormap
colorbar; % Show colorbar
caxis([min_temp, max_temp]); % Set the color axis scaling to match the temperature range

% Overlay the boundary
hold on;
scatter(stations(:, 1), stations(:, 2), 'magenta', 'filled');
plot(boundaryMat(:,1), boundaryMat(:,2), 'k-', 'LineWidth', 0.2); % Plot the boundary in black

% Add labels and title to the plot
xlabel('Longitude');
ylabel('Latitude');
title('f) January (c - a)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, 'January (c - a).jpg', '-djpeg', '-r300');