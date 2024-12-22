%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ERE621 - Spatial Analysis
% Taposh Mollick
% Class project
% Ordinary Kriging for Temperature prediction using Bangladesh BMD Data
% Tmax_2008_2017_Winter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc

% Load the dataset
load('Bangladesh_BMD_Tmax_35_years.mat');
load("GIS_interpolated_1984_2017.mat");

InMat = Tmax;

InMat2 = BMD_Stations_Elevation;
X = InMat2(:, 1);
Y = InMat2(:, 2);
height = InMat2(:, 3);

outer = BGD_outer_boundary;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the 35 BMD weather stations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boundaryMat = BGD_division_boundary;

stations = BMD_Stations_Elevation;

% Plot the boundary
figure; % Create a new figure
scatter(stations(:, 1), stations(:, 2), 'red', 'filled');
hold on
plot(boundaryMat(:,1), boundaryMat(:,2), 'b-', 'LineWidth', 0.5); % Plot as a blue line with a thickness of 2
xlabel('Longitude'); % Label the x-axis
ylabel('Latitude'); % Label the y-axis
title('Meteorological Stations, Bangladesh'); % Title for the plot
grid on; % Turn on the grid
axis equal; % Set the aspect ratio so that the data units are the same in every direction
hold off

%print(gcf, 'Study_area_Meteorological_stations.jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate distances among the BMD weather station
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

distanceMatrix = calculateDistanceMatrix(InMat2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Replace the NaN values, less than 5 values and greater than 45 values of 
% Tmax dataset by nearest weather station data of that day
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize the new matrix
Tmax_new = InMat;

% Number of weather stations
numStations = size(InMat, 2) - 3;

% Process each weather station column in InMat
for stationIdx = 4:(3 + numStations)
    % Iterate through each day in the dataset
    for idx = 1:size(InMat, 1)
        currentValue = InMat(idx, stationIdx);
        
        % Check for NaN or out-of-range values
        if isnan(currentValue) || currentValue < 5 || currentValue > 45
            % Sort the distances for the current station
            [sortedDistances, sortedIndices] = sort(distanceMatrix(stationIdx - 3, :), 'ascend');

            % Find the nearest station with valid data
            for nearestStationOffset = sortedIndices
                nearestStationIdx = nearestStationOffset + 3; % Adjusting index for station columns
                
                % Avoid using the same station
                if nearestStationIdx ~= stationIdx
                    nearestStationValue = InMat(idx, nearestStationIdx);
                    
                    % Check if the nearest station value is valid
                    if ~isnan(nearestStationValue) && nearestStationValue >= 5 && nearestStationValue <= 45
                        % Replace the invalid value with the value from the nearest station
                        Tmax_new(idx, stationIdx) = nearestStationValue;
                        break; % Exit the loop once a valid value is found
                    end
                end
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate monthly average temperature of Tmax_new datast
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of weather stations
numStations = size(Tmax_new, 2) - 3;

% Extract Year and Month columns
Year = Tmax_new(:, 1);
Month = Tmax_new(:, 2);

% Find unique years and months
uniqueYears = unique(Year);
uniqueMonths = 1:12; % Months from January to December

% Initialize the matrix to store monthly averages
monthly_mean = NaN(length(uniqueYears) * length(uniqueMonths), numStations + 2);

% Populate Year and Month in the monthly_mean matrix
for yIdx = 1:length(uniqueYears)
    for mIdx = 1:length(uniqueMonths)
        rowIdx = (yIdx - 1) * length(uniqueMonths) + mIdx;
        monthly_mean(rowIdx, 1) = uniqueYears(yIdx);
        monthly_mean(rowIdx, 2) = uniqueMonths(mIdx);
    end
end

% Loop over each weather station column
for stationIdx = 1:numStations
    % Loop over each month
    for mIdx = 1:length(uniqueMonths)
        % Loop over each year
        for yIdx = 1:length(uniqueYears)
            % Find indices for the current month and year
            indices = (Year == uniqueYears(yIdx)) & (Month == uniqueMonths(mIdx));

            % Extract the data for the current month, year, and station
            monthlyData = Tmax_new(indices, stationIdx + 3);

            % Filter out any non-positive values to ensure accuracy
            monthlyData = monthlyData(monthlyData > 0);

            % Calculate the mean for the current month and year
            if ~isempty(monthlyData)
                monthlyMean = mean(monthlyData, 'omitnan');
            else
                monthlyMean = NaN; % If there are no valid data points, assign NaN
            end

            % Find the corresponding row in the monthly_mean matrix
            rowIdx = (yIdx - 1) * length(uniqueMonths) + mIdx;

            % Assign the mean to the monthly_mean matrix
            monthly_mean(rowIdx, stationIdx + 2) = monthlyMean;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transform the dataset into log normal and make new dataset name
% Tmax_final
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize 'Tmax_final' with the same size as 'monthly_mean'
Tmax_final = monthly_mean;

% Loop through each year from 1984 to 2017
for year = 1984:2017
    % Loop through each month
    for month = 1:12
        % Find the rows corresponding to the current year and month
        rows = monthly_mean(:,1) == year & monthly_mean(:,2) == month;
        
        % Apply log transformation to columns 3 to 37
        Tmax_final(rows, 3:37) = log(monthly_mean(rows, 3:37));
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Split the dataset into 3 segments
% 1984-1995 = 12 years
% 1996-2007 = 12 Years
% 2008-2018 = 11 Years
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Splitting the Tmax_final dataset into three segments based on years
Year = Tmax_final(:, 1);
Tmax_1 = Tmax_final(Year >= 1984 & Year <= 1995, :);
Tmax_2 = Tmax_final(Year >= 1996 & Year <= 2007, :);
Tmax_3 = Tmax_final(Year >= 2008 & Year <= 2017, :);

% Define winter and summer months
winter_months = [11, 12, 1, 2]; % November to February
summer_months = [4, 5, 6, 7];   % June to September

% Process each segment and split into winter and summer datasets
[Tmax_1_coord] = processAndVisualizeSegment(Tmax_1, X, Y);
[Tmax_2_coord] = processAndVisualizeSegment(Tmax_2, X, Y);
[Tmax_3_coord] = processAndVisualizeSegment(Tmax_3, X, Y);

% Filter out winter and summer months for each dataset
Tmax_1_winter = filterForMonths(Tmax_1_coord, winter_months);
Tmax_1_summer = filterForMonths(Tmax_1_coord, summer_months);

Tmax_2_winter = filterForMonths(Tmax_2_coord, winter_months);
Tmax_2_summer = filterForMonths(Tmax_2_coord, summer_months);

Tmax_3_winter = filterForMonths(Tmax_3_coord, winter_months);
Tmax_3_summer = filterForMonths(Tmax_3_coord, summer_months);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Histogram and normal distribution curve for a selected period
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Year, Month, and Temperature columns for all stations
Year = monthly_mean(:, 1);
Month = monthly_mean(:, 2);
Temps = monthly_mean(:, 3:37); % Temperature data from columns 3 to 37

% Filter to include only data for a specified period
specifiedPeriodIndices = Year == 1984 & Month == 1; % Logical indices for the specified period
specifiedPeriodTemps = Temps(specifiedPeriodIndices, :); % All temperature columns for the specified period

% Flatten the matrix to get a single vector of all temperature values for the specified period
specifiedPeriodTemps = specifiedPeriodTemps(:);

% Calculate mean and standard deviation for the temperatures
mu = mean(specifiedPeriodTemps, 'omitnan');
sigma = std(specifiedPeriodTemps, 'omitnan');

% Create histogram of the data
figure;
histogram(specifiedPeriodTemps, 5, 'Normalization', 'pdf'); % Specify number of bins directly

hold on; % Keep the histogram on the plot

% Generate values for the normal distribution curve
x = linspace(min(specifiedPeriodTemps), max(specifiedPeriodTemps), 100);
y = normpdf(x, mu, sigma);

% Plot the normal distribution curve
plot(x, y, 'LineWidth', 2, 'Color', 'red');

% Label the plot
xlabel('Temperature (°C)');
ylabel('Probability Density');
title('a) Original Data');
legend('Histogram', 'Normal Distribution');

hold off; % Release the plot hold
%print(gcf, 'Histogram of Original Data.jpg', '-djpeg', '-r300');


% Log Transformed data

% Extract Year, Month, and Temperature columns for all stations
Year = Tmax_final(:, 1);
Month = Tmax_final(:, 2);
Temps = Tmax_final(:, 3:37); % Temperature data from columns 3 to 37

% Filter to include only data for a specified period
specifiedPeriodIndices = Year == 1984 & Month == 1; % Logical indices for the specified period
specifiedPeriodTemps = Temps(specifiedPeriodIndices, :); % All temperature columns for the specified period

% Flatten the matrix to get a single vector of all temperature values for the specified period
specifiedPeriodTemps = specifiedPeriodTemps(:);

% Calculate mean and standard deviation for the temperatures
mu = mean(specifiedPeriodTemps, 'omitnan');
sigma = std(specifiedPeriodTemps, 'omitnan');

% Create histogram of the data
figure;
histogram(specifiedPeriodTemps, 5, 'Normalization', 'pdf'); % Specify number of bins directly

hold on; % Keep the histogram on the plot

% Generate values for the normal distribution curve
x = linspace(min(specifiedPeriodTemps), max(specifiedPeriodTemps), 100);
y = normpdf(x, mu, sigma);

% Plot the normal distribution curve
plot(x, y, 'LineWidth', 2, 'Color', 'red');

% Label the plot
xlabel('Temperature');
ylabel('Probability Density');
title('b) Log Transformed Data');
legend('Histogram', 'Normal Distribution');

hold off; % Release the plot hold
%print(gcf, 'Histogram of Log Transformed Data.jpg', '-djpeg', '-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variogram calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numbin = 40;
w = 1;

Mat = Tmax_3_winter;
X = Mat(1, 3:end)';
Y = Mat(2, 3:end)';
Z = Mat(6, 3:end)';
cal_var = var(Z);

XYZ = [X, Y, Z];

% Variogram parameters
hmax = 401.8342;
num = 15;
lag = hmax / (2*(num-1)+1);
cloud_number = 0;

% Best fit parameters for different models
% Spherical
sill1 = 0.00610037;
nugget1 = 0.000681281;
range1 = 480;

%Exponential
sill2 = 0.00478953;
nugget2 = 0.000681281;
range2 = 460;

% Gaussian
sill3 = 0.00630037;
nugget3 = 0.000940777;
range3 = 450.832;

% Linear
sill4 = 0.00461337;
nugget4 = 0.000681281;
range4 = 346.409;

% Calculate empirical variogram
[h, rh] = decide_variogram(XYZ, hmax, num, lag, cloud_number, sill1, range1, nugget1);

% Extended range for theoretical models
extended_hk = linspace(0, hmax, 100);  

% Theoretical models
Gamma_spherical = nugget1 + ((sill1 - nugget1) * (((3*extended_hk) ./ (2*range1)) - ((extended_hk.^3)./(2*range1.^3))));
Gamma_exponential = nugget2 + ((sill2 - nugget2) * (1 - exp(-3*(extended_hk/range2))));
Gamma_gaussian = nugget3 + ((sill3 - nugget3) * min((1 - exp(-3*(extended_hk/range3).^2)), 1));
Gamma_linear = min(nugget4 + (sill4 - nugget4) * extended_hk / range4, sill4);

% Plotting Empirical Variogram
figure;
plot(h, rh, '*r', 'MarkerSize', 10); hold on;
title('Empirical Variogram');
xlabel('Distance (h)');
ylabel('Semivariance');
grid on;
%legend('Empirical', 'Spherical', 'Location', 'northwest');
%print(gcf, 'Empirical Variogram.jpg', '-djpeg', '-r300');

% Plotting Spherical Model
figure;
plot(h, rh, '*r', 'MarkerSize', 10); hold on;
plot(extended_hk, Gamma_spherical, '-b', 'LineWidth', 1);
title('Spherical Model');
xlabel('Distance (h)');
ylabel('Semivariance');
grid on;
%legend('Empirical', 'Spherical', 'Location', 'northwest');
%print(gcf, 'Spherical Model.jpg', '-djpeg', '-r300');

% Plotting Exponential Model
figure;
plot(h, rh, '*r', 'MarkerSize', 10); hold on;
plot(extended_hk, Gamma_exponential, '-g', 'LineWidth', 1);
title('Exponential Model');
xlabel('Distance (h)');
ylabel('Semivariance');
grid on;
%legend('Empirical', 'Exponential', 'Location', 'northwest');
%print(gcf, 'Exponential Exponential.jpg', '-djpeg', '-r300');

% Plotting Gaussian Model
figure;
plot(h, rh, '*r', 'MarkerSize', 10); hold on;
plot(extended_hk, Gamma_gaussian, '-m', 'LineWidth', 1);
title('Gaussian Model');
xlabel('Distance (h)');
ylabel('Semivariance');
grid on;
%legend('Empirical', 'Gaussian', 'Location', 'northwest');
%print(gcf, 'Gaussian Model.jpg', '-djpeg', '-r300');

% Plotting Linear Model
figure;
plot(h, rh, '*r', 'MarkerSize', 10); hold on;
plot(extended_hk, Gamma_linear, '-k', 'LineWidth', 1);
title('Linear Model');
xlabel('Distance (h)');
ylabel('Semivariance');
grid on;
%legend('Empirical', 'Linear', 'Location', 'northwest');
%print(gcf, 'Linear Model.jpg', '-djpeg', '-r300');

colormap(gca, cool);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Application of ordinary kriging functions 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Data preparation from Tmax_1_summer
Mat = Tmax_3_winter;
X_station = Mat(1, 3:end)';  % Longitude of stations
Y_station = Mat(2, 3:end)';  % Latitude of stations
Z = Mat(3:end, 3:end)';  % Temperature data, transposed to have columns as different datasets

% Define the boundary from boundaryMat
boundaryLong = boundaryMat(:, 1);
boundaryLat = boundaryMat(:, 2);

% Create a denser grid of points
n = 20; % Adjust the density of the grid as needed
ti_x = linspace(min(boundaryLong), max(boundaryLong), n);
ti_y = linspace(min(boundaryLat), max(boundaryLat), n);
[X2, Y2] = meshgrid(ti_x, ti_y);

% Flatten the grid to a list of points
Matp_all = [X2(:) Y2(:)];  

% Filter points to keep only those within the boundary
[inBoundary, ~] = inpolygon(Matp_all(:, 1), Matp_all(:, 2), boundaryLong, boundaryLat);
Matp = Matp_all(inBoundary, :);  % Prediction points within the boundary

% Now Matp contains the prediction points within and slightly beyond the boundary

% Variogram calculation parameters
sill = 0.00610037;
nugget = 0.000681281;
range = 480;
cov_function_shape = 0;  % Assuming spherical model

% Initialize matrices to store kriging results for each column in Z
num_columns = size(Z, 2);  % Number of columns in Z
num_predictions = size(Matp, 1);  % Number of prediction points
Zpr_1_summer_or = zeros(num_predictions, num_columns);
Zpr_error_1_summer_or = zeros(num_predictions, num_columns);

% Loop through each column of Z
for col = 1:num_columns
    % Extract the temperature data for the current column
    Z_col = Z(:, col);
    sigma_sq = var(Z_col, 'omitnan');  % Use 'omitnan' to ignore NaNs

    % Combine station coordinates with temperature data
    XYZ = [X_station, Y_station, Z_col];

    % Perform ordinary kriging for the current column
    [Zpr_col, Zpr_error_col, h, variogram, COV1] = Mollick_Ordinary_kriging(XYZ, Matp, sigma_sq, sill, range, nugget, cov_function_shape);

    % Store kriging results for the current column
    Zpr_1_summer_or(:, col) = Zpr_col;
    Zpr_error_1_summer_or(:, col) = Zpr_error_col;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate mean temperature of January for Map generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract longitude and latitude
longitude = winter2008pr(:, 1);
latitude = winter2008pr(:, 2);

% Extract temperature data from column 3, 7, 11, ... (January months)
num_columns = size(winter2008pr, 2);
january_indices = 3:4:num_columns;

% Calculate the mean of all April months
mean_january_temps = mean(winter2008pr(:, january_indices), 2);

% Combine longitude, latitude, and mean January temperatures into a new matrix
winter2008prtemp = [longitude, latitude, mean_january_temps];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate predicted map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pr2008 = winter2008prtemp;

% Extract the 45th Column of Zpr_1_summer_or as temperature predictions
temp_pr = pr2008(:, 3);

% Extract Longitude and Latitude from Matp
longitudes = pr2008(:, 1);
latitudes = pr2008(:, 2);

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
title('a) January (2008 - 2017)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Predicted Maximum Temperature of January - 1984 - 1995.jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate mean error of January for Map generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract longitude and latitude
longitude = winter2008err(:, 1);
latitude = winter2008err(:, 2);

% Extract error data from column 3, 7, 11, ... (January months)
num_columns = size(winter2008err, 2);
january_indices = 3:4:num_columns;

% Calculate the mean of all April months
mean_january_error = mean(winter2008err(:, january_indices), 2);

% Combine longitude, latitude, and mean January temperatures into a new matrix
winter2008prerr = [longitude, latitude, mean_january_error];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate error map or figures for Ordinary Kriging prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err2008 = winter2008prerr;
% Extract the 45 Column of Zpr_1_summer_or as temperature predictions
temp_pr = err2008(:, 3);

% Extract Longitude and Latitude from Matp
longitudes = err2008(:, 1);
latitudes = err2008(:, 2);

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
title('a) January (2008 - 2017)');

% Set axis limits to the boundary data range
xlim([min(boundaryMat(:,1)), max(boundaryMat(:,1))]);
ylim([min(boundaryMat(:,2)), max(boundaryMat(:,2))]);
axis equal; % Equal scaling on both axes

% Release the plot hold
hold off;
print(gcf, ' Error Map of January - 2008 - 2017.jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate RMSE and R-squared values for ordinary Kriging
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assuming Z_actual contains the actual values and is of the same size as Zpr_error_1_summer_or
% Extracting the first column of actual values and error values
actual_values = winter2008prtemp(:, 3);
error_values = winter2008prerr(:, 3);

% Calculate the predicted values
predicted_values = actual_values - error_values;

% Calculate RMSE
squared_errors = error_values .^ 2;
rmse = sqrt(mean(squared_errors));

% Display RMSE
disp(['RMSE for Ordinary: ', num2str(rmse)]);

% Calculate R-squared
% Total Sum of Squares (TSS)
mean_actual = mean(actual_values);
tss = sum((actual_values - mean_actual) .^ 2);

% Residual Sum of Squares (RSS)
rss = sum(squared_errors);

% R-squared
r_squared = 1 - (rss / tss);

% Display R-squared
disp(['R-squared for Ordinary: ', num2str(r_squared)]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cross-Validation of Ordinary Kriging
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(123457) % Set seed to get same R-squared and same RMSE every time 

numbin = 22;
w=1;

Mat = Tmax_3_winter;

X = Mat(1, 3:end)';  % Longitude of stations
Y = Mat(2, 3:end)';  % Latitude of stations
Z = Mat(6, 3:end)';  % Temperature data, transposed to have columns as different datasets
cal_var = var(Z);

XYZ = [X, Y, Z];
hmax = 401.8342;
num = 22;
lag = hmax / (2*(num-1)+1);
cloud_number = 0;

% Best fit
sill = 0.00610037;
nugget = 0.000681281;
range = 480;
cov_function_shape = 0;
sigma_sq = sill;


% Number of folds for cross-validation
num_folds = 5;
% Get the number of data points
num_points = size(XYZ, 1);
% Randomly shuffle the indices
shuffled_indices = randperm(num_points);
% Determine the number of points in each fold
fold_size = floor(num_points / num_folds);
% Preallocate arrays to store cross-validation results
Zpr_cv = zeros(num_points, 1);
Zpr_error_cv = zeros(num_points, 1);
% Perform cross-validation
for i = 1:num_folds
    % Define the indices for the current fold
    fold_indices = shuffled_indices((i - 1) * fold_size + 1 : min(i * fold_size, num_points));
    
    % Create training and testing sets
    training_set = XYZ(setdiff(1:num_points, fold_indices), :);
    testing_set = XYZ(fold_indices, :);
    
    % Perform Ordinary Kriging on the training set
    [Zpr_cv(fold_indices), Zpr_error_cv(fold_indices), h, variogram, COV1] = Mollick_Ordinary_kriging(training_set, testing_set, sigma_sq, sill, range, nugget, cov_function_shape);
end

error = exp(Z)- exp(Zpr_cv);
squared_errors = error .^ 2;
mean_squared_error = mean(squared_errors);
rmse = sqrt(mean_squared_error); % Use antilog because the original data was transformed into log
disp(['RSME value for cross validation is: ', num2str(rmse)]);

% Calculate Total Sum of Squares (TSS)
Z_mean = mean(exp(Z));
total_variance = (exp(Z) - Z_mean) .^ 2;
total_sum_of_squares = sum(total_variance);

% Calculate Residual Sum of Squares (RSS)
residual_sum_of_squares = sum(squared_errors);

% Calculate R-squared value
r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares);

% Display the R-squared value
disp(['R-squared value for cross validation is: ', num2str(r_squared)]);

w=w+1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate figure for Cross-Validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Z = exp(Z);
Zpr_cv = exp(Zpr_cv);
% Fit a line to the data
p = polyfit(Z, Zpr_cv, 1);
% Create a vector of x values for plotting the fit line
x_fit = linspace(min(Z), max(Z), 100);
% Generate y values based on the fit
y_fit = polyval(p, x_fit);

% Create the scatter plot
figure;
scatter(Z, Zpr_cv, 'filled');
hold on;
% Plot the fit line
plot(x_fit, y_fit, 'r-', 'LineWidth', 2);

% Get upper limit of y-axis for positioning text
y_upper_limit = max(get(gca, 'ylim'));

% Offset for placing text below the margin line
offset = (y_upper_limit - min(Zpr_cv)) * 0.05; % Adjust as needed

% Annotate the plot with the line equation, R-squared, and RMSE
% Position the annotations just below the upper margin
text(min(Z), y_upper_limit - offset, sprintf('y = %.2fx + %.2f', p(1), p(2)), ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 10);
text(min(Z), y_upper_limit - 2 * offset, sprintf('R-squared = %.2f', r_squared), ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 10);
text(min(Z), y_upper_limit - 3 * offset, sprintf('RMSE = %.2f', rmse), ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 10);

% Set the axis labels
xlabel('Actual Temperature');
ylabel('Predicted Temperature');
title('f) January (2008 - 2017)');

% Add grid for better readability
grid on;
hold off;
print(gcf, 'Cross Validation Actual vs Predicted Temperature of January (2008 - 2017).jpg', '-djpeg', '-r300');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time series plot using best predicted temperatures values beased on
% R-squared and RMSE error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assuming Predicted_temp is your data matrix with temperature predictions
Predicted_temp = winter2008pr(:, 3:end); % Replace with your actual data

% Pre-allocate arrays for storing average temperatures for November to February
numYears = 10; % From 1996 to 2007
avgTempJan = zeros(numYears, 1);
avgTempFeb = zeros(numYears, 1);
avgTempNov = zeros(numYears, 1);
avgTempDec = zeros(numYears, 1);
%avgTempApril = zeros(numYears, 1);
%avgTempMay = zeros(numYears, 1);
%avgTempJune = zeros(numYears, 1);
%avgTempJuly = zeros(numYears, 1);

% Loop through each year to calculate average temperatures
for i = 1:numYears
    colStart = (i - 1) * 4 + 1; % Starting column for November of the current year
    avgTempJan(i) = mean(Predicted_temp(:, colStart), 'omitnan');
    avgTempFeb(i) = mean(Predicted_temp(:, colStart + 1), 'omitnan');
    avgTempNov(i) = mean(Predicted_temp(:, colStart + 2), 'omitnan');
    avgTempDec(i) = mean(Predicted_temp(:, colStart + 3), 'omitnan');
end
%for i = 1:numYears
    %colStart = (i - 1) * 4 + 1; % Starting column for the current year
    %avgTempApril(i) = mean(Predicted_temp(:, colStart), 'omitnan');
    %avgTempMay(i) = mean(Predicted_temp(:, colStart + 1), 'omitnan');
    %avgTempJune(i) = mean(Predicted_temp(:, colStart + 2), 'omitnan');
    %avgTempJuly(i) = mean(Predicted_temp(:, colStart + 3), 'omitnan');
%end
% Define years from 1984 to 1995
years = 2008:2017;

% Plotting
figure;
plot(years, avgTempJan, '-o', 'color', 'blue', 'linewidth', 1.5, 'DisplayName', 'January');
hold on;
plot(years, avgTempFeb, '-o', 'color', 'cyan', 'linewidth', 1.5, 'DisplayName', 'February');
plot(years, avgTempNov, '-o', 'color', 'magenta', 'linewidth', 1.5, 'DisplayName', 'November');
plot(years, avgTempDec, '-o', 'color', 'green', 'linewidth', 1.5, 'DisplayName', 'December');

%plot(years, avgTempApril, '-o', 'color', 'red', 'linewidth', 1.5, 'DisplayName', 'April');
%hold on;
%plot(years, avgTempMay, '-o', 'color', 'magenta', 'linewidth', 1.5, 'DisplayName', 'May');
%plot(years, avgTempJune, '-o', 'color', 'green', 'linewidth', 1.5, 'DisplayName', 'June');
%plot(years, avgTempJuly, '-o', 'color', 'blue', 'linewidth', 1.5, 'DisplayName', 'July');
hold off;

% Adding labels, title, and legend
xlabel('Year');
ylabel('Temperature (°C)');
title('f) Winter (2008 - 2017)');
legend('Location', 'southeast', 'FontSize', 8); % Smaller font size for the legend
legend boxoff; % Remove box around the legend
grid on;
print(gcf, 'Timeseries Average Maximum Temperatures of Winter (2008 - 2017).jpg', '-djpeg', '-r300');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to add coordinate with each segment (Tmax_1, Tmax_2, Tmax_3) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Tmax_coord] = processAndVisualizeSegment(Tmax_segment, X, Y)
    % Add Longitude and Latitude
    new_rows = nan(2, size(Tmax_segment, 2));
    new_rows(1, 3:end) = X';
    new_rows(2, 3:end) = Y';
    Tmax_coord = [new_rows; Tmax_segment];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to filter months and split into training and test sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function to filter the Tmax dataset for specified months
function Tmax_filtered = filterForMonths(Tmax, months)
    % Filter rows where the second column (Month) matches the specified months
    filtered_data = Tmax(ismember(Tmax(:, 2), months), :);
    
    % Preserve the first two rows (Longitude and Latitude)
    first_two_rows = Tmax(1:2, :);
    
    % Combine the preserved rows with the filtered data
    Tmax_filtered = [first_two_rows; filtered_data];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mollick_ordinary_kriging function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Zpr_or, Zpr_error_or, h,variogram, COV1] = Mollick_Ordinary_kriging(XYZ, Matp, sigma_sq, sill, range, nugget, cov_function_shape)
    % n×3 matrix containing the X, Y and Z(attribute) coordinates of observation points
    X = XYZ(:,1);
    Y = XYZ(:,2);
    Z = XYZ(:,3);
    % Calculating a nxn distance matrix of the observation points
    n = size(XYZ,1);
    %distance = zeros(n, n);
   % for i = 1:n
      %  for j = 1:n
         %   dx = XYZ(i,1)-XYZ(j,1);
          %  dy = XYZ(i,2)-XYZ(j,2);
         %   distance(i,j)= sqrt(dx^2 + dy^2);
       % end
    %end
    distance = calculateDistanceMatrix(XYZ);
    % Calculating variogram
    variogram = zeros(n, n);
    for i = 1:n
        for j = 1:n
            h = distance(i, j);
            variogram(i, j) = compute_variogram(h, nugget, sill, range, cov_function_shape);
        end
    end

    % Converting distance matrix into covariances
    COV1 = sigma_sq - variogram;
    % Initialize the expanded matrix with zeros
    OK_COV1 = zeros(n+1, n+1);
    % Single for loop to fill in the values
    for i = 1:(n+1)
        for j = 1:(n+1)
            if i <= n && j <= n
                % Inside the original matrix dimensions, copy the old values
                OK_COV1(i, j) = COV1(i, j);
            elseif i == (n+1) && j == (n+1)
                % Bottom-right corner should be zero
                OK_COV1(i, j) = 0;
            else
                % Last row and last column should be one
                OK_COV1(i, j) = 1;
            end
        end
    end

    % Inverse of covariance matrix
    OK_COV1_inv = inv(OK_COV1);
    
    k = size(Matp,1); % Number of prediction points
    Zpr_or = zeros(k,1);
    Zpr_error_or = zeros(k,1);
    
    
        % Earth's mean radius in kilometers
        size(Matp,1);
        size(XYZ,1);
    R = 6371;
    for i = 1:k
        Xp = Matp(i, 1);
        Yp = Matp(i, 2);
            lat1 = deg2rad(X);
            lon1 = deg2rad(Y);
            lat2 = deg2rad(Xp);
            lon2 = deg2rad(Yp);

            dLat = lat2 - lat1;
            dLon = lon2 - lon1;

            % Haversine formula
            a = sin(dLat/2).^2 + cos(lat1) .* cos(lat2) .* sin(dLon/2).^2;
            c = 2 * atan2(sqrt(a), sqrt(1-a));
            d = R * c;

            distance_pred = d;
    
        % Predicting at Matp(i,:)
        variogram_p = zeros(n, 1);
        for j = 1:n
            h = distance_pred(j);
            variogram_p(j) = compute_variogram(h, nugget, sill, range, cov_function_shape);
        end
        % Use C(h) to convert the distance matrix into covariances
        OK_c_pred = [sigma_sq - variogram_p; 1];
        % Calculate weights using the ordinary kriging system
        omega = OK_COV1_inv * OK_c_pred; 
        % Calculate the predicted Z value
        Zpr_or(i) = XYZ(:,3)' * omega(1:end-1);   
        % Calculate the kriging variance
        Zpr_error_or(i) = sigma_sq - (omega' * OK_c_pred);

    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decide_Variogram function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [h, rh] = decide_variogram(XYZ, hmax, num, lag, cloud_number, sill, range, nugget)
    X = XYZ(:,1);
    Y = XYZ(:,2);
    Z = XYZ(:,3);
    n = size(XYZ, 1);
    count = zeros(1, n);
    sumvar = zeros(1, n);
    h = lag:2*lag:hmax;
    rh = zeros(1, num);
    %distmat = zeros(n, n);
    distmat = calculateDistanceMatrix(XYZ);
    %for i = 1:n
        %for j = 1:n
            %distmat(i, j) = sqrt((X(i) - X(j))^2 + (Y(i) - Y(j))^2);
        %end
    %end
    for m = 1:num
        sumvar = 0;
        for k = 1:n
            ind = find(distmat(:,k) > (h(m) - lag) & distmat(:,k) <= (h(m) + lag));
            count(k) = numel(ind);
            if count(k) > 0
                diffs = (Z(ind) - Z(k)).^2;
                sumvar = sumvar + sum(diffs);
                if cloud_number == 1
                    plot(h(m), diffs, '.');
                    hold on;
                end
            end
        end
        total_pairs = sum(count);
        if total_pairs > 0
            rh(m) = sumvar / (2 * total_pairs);
        else
            rh(m) = 0;
        end
    end
    if cloud_number == 0
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculateDiatanceMatrix function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function distanceMatrix = calculateDistanceMatrix(InMat)
    % Earth's mean radius in kilometers
    R = 6371;
    n = size(InMat, 1);
    distanceMatrix = zeros(n, n);

    for i = 1:n
        for j = 1:n
            lat1 = deg2rad(InMat(i, 1));
            lon1 = deg2rad(InMat(i, 2));
            lat2 = deg2rad(InMat(j, 1));
            lon2 = deg2rad(InMat(j, 2));

            dLat = lat2 - lat1;
            dLon = lon2 - lon1;

            % Haversine formula
            a = sin(dLat/2)^2 + cos(lat1) * cos(lat2) * sin(dLon/2)^2;
            c = 2 * atan2(sqrt(a), sqrt(1-a));
            d = R * c;

            distanceMatrix(i, j) = d;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the function for variogram calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gamma = compute_variogram(h, nugget, sill, range, cov_function_shape)
    if cov_function_shape == 0 % Spherical model
        if h == 0
            gamma = 0;
        elseif h <= range
            gamma = nugget + (sill - nugget) * (((3/2) * (h / range)) - (1/2) * (h.^3 / range.^3));
        else
            gamma = sill;
        end
    elseif cov_function_shape == 1 % Exponential model
        if h > 0
            gamma = nugget + (sill - nugget) * (1 - exp(-3 * h / range));
        else
            gamma = 0;
        end
    elseif cov_function_shape == 2 % Gaussian model
        gamma = nugget + ((sill - nugget) * min((1 - exp(-3 * (h/range).^2)), 1));
    else
        error('Invalid cov_function_shape');
    end
end





