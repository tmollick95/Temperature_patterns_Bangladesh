% Load dataset
load("Interpolated_and_error_final_maps.mat");
load('Bangladesh_BMD_Tmax_35_years.mat');
load("GIS_interpolated_1984_2017.mat");

boundaryMat = BGD_division_boundary;

stations = BMD_Stations_Elevation;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate predicted map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pr1984 = Interpolated;

% Extract the 45th Column of Zpr_1_summer_or as temperature predictions
temp_pr = pr1984(:, 3);

% Extract Longitude and Latitude from Matp
longitudes = pr1984(:, 1);
latitudes = pr1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('a) January 2022 (BMD)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Predicted Maximum Temperature of January 2022 (BMD).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate error map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err1984 = error;
% Extract the 45 Column of Zpr_1_summer_or as temperature predictions
temp_pr = err1984(:, 3);

% Extract Longitude and Latitude from Matp
longitudes = err1984(:, 1);
latitudes = err1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('a) January 2022 (BMD Error)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Error Map of January 2022 (BMD Error).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate predicted map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pr1984 = Interpolated;

% Extract the 45th Column of Zpr_1_summer_or as temperature predictions
temp_pr = pr1984(:, 4);

% Extract Longitude and Latitude from Matp
longitudes = pr1984(:, 1);
latitudes = pr1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('b) January 2022 (Predicted)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Predicted Maximum Temperature of January 2022 (Predicted).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate error map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err1984 = error;
% Extract the 45 Column of Zpr_1_summer_or as temperature predictions
temp_pr = err1984(:, 4);

% Extract Longitude and Latitude from Matp
longitudes = err1984(:, 1);
latitudes = err1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('b) January 2022 (Predicted Error)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Error Map of January 2022 (Predicted Error).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate predicted map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pr1984 = Interpolated;

% Extract the 45th Column of Zpr_1_summer_or as temperature predictions
temp_pr = pr1984(:, 5);

% Extract Longitude and Latitude from Matp
longitudes = pr1984(:, 1);
latitudes = pr1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('c) April 2022 (BMD)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Predicted Maximum Temperature of April 2022 (BMD).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate error map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err1984 = error;
% Extract the 45 Column of Zpr_1_summer_or as temperature predictions
temp_pr = err1984(:, 5);

% Extract Longitude and Latitude from Matp
longitudes = err1984(:, 1);
latitudes = err1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('c) April 2022 (BMD Error)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Error Map of April 2022 (BMD Error).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate predicted map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pr1984 = Interpolated;

% Extract the 45th Column of Zpr_1_summer_or as temperature predictions
temp_pr = pr1984(:, 6);

% Extract Longitude and Latitude from Matp
longitudes = pr1984(:, 1);
latitudes = pr1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('d) April 2022 (Predicted)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Predicted Maximum Temperature of April 2022 (Predicted).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate error map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err1984 = error;
% Extract the 45 Column of Zpr_1_summer_or as temperature predictions
temp_pr = err1984(:, 6);

% Extract Longitude and Latitude from Matp
longitudes = err1984(:, 1);
latitudes = err1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('d) April 2022 (Predicted Error)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Error Map of April 2022 (Predicted Error).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate predicted map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pr1984 = Interpolated;

% Extract the 45th Column of Zpr_1_summer_or as temperature predictions
temp_pr = pr1984(:, 7);

% Extract Longitude and Latitude from Matp
longitudes = pr1984(:, 1);
latitudes = pr1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('e) January 2027 (BMD)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Predicted Maximum Temperature of January 2027 (BMD).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate error map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err1984 = error;
% Extract the 45 Column of Zpr_1_summer_or as temperature predictions
temp_pr = err1984(:, 7);

% Extract Longitude and Latitude from Matp
longitudes = err1984(:, 1);
latitudes = err1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('e) January 2027 (BMD Error)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Error Map of January 2027 (BMD Error).jpg', '-djpeg', '-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate predicted map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pr1984 = Interpolated;

% Extract the 45th Column of Zpr_1_summer_or as temperature predictions
temp_pr = pr1984(:, 8);

% Extract Longitude and Latitude from Matp
longitudes = pr1984(:, 1);
latitudes = pr1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('f) April 2027 (Predicted)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Predicted Maximum Temperature of April 2027 (Predicted).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate error map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err1984 = error;
% Extract the 45 Column of Zpr_1_summer_or as temperature predictions
temp_pr = err1984(:, 8);

% Extract Longitude and Latitude from Matp
longitudes = err1984(:, 1);
latitudes = err1984(:, 2);

% Define an extended grid for the contour plot beyond the boundary limits
padding = 0.01 * (max(boundaryMat(:,1)) - min(boundaryMat(:,1))); % padding for the grid
long_range = linspace(min(boundaryMat(:,1)) - padding, max(boundaryMat(:,1)) + padding, 300);
lat_range = linspace(min(boundaryMat(:,2)) - padding, max(boundaryMat(:,2)) + padding, 300);
[long_grid, lat_grid] = meshgrid(long_range, lat_range);

% Interpolate the temperature data onto the grid
% Use 'linear' if 'natural' does not extrapolate well to the edges
F = scatteredInterpolant(longitudes, latitudes, temp_pr, 'linear', 'linear');
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
title('f) April 2027 (Predicted Error)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Error Map of April 2027 (Predicted Error).jpg', '-djpeg', '-r300');