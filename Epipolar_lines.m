im1 = rgb2gray(imread("91968\Desktop\RoboticVision\project_data (1)\image_left_unrect\image_left_unrect_2.jpg"));
im2 = rgb2gray(imread("91968\Desktop\RoboticVision\project_data (1)\image_right_unrect\image_right_unrect_2.jpg"));
rx = 0.0034;
ry = 0.0000;
rz = -0.0008;
rotationVector = [rx, ry, rz];

% Compute the angle of rotation (the magnitude of the rotation vector)
theta = norm(rotationVector);

% Normalize the rotation vector to get the rotation axis
rotationAxis = rotationVector / theta;

% Create the rotation vector in the format [axis, angle]
rotationVecWithAngle = [rotationAxis, theta];

% Convert the rotation vector to a rotation matrix
R = [1.000,0.0008,-0.0000;-0.0008,1.0000,-0.0034;-0.0000,0.0034,1.0000];

% Example intrinsic parameters (replace with actual values from calibration)
fx1 = 1401.1400; fy1 = 1401.1400; cx1 = 1150.3900; cy1 = 670.6210; % Intrinsic parameters for left camera
fx2 = 1399.2200; fy2 = 1399.2200; cx2 = 1115.0000; cy2 = 600.0290; % Intrinsic parameters for right camera

leftCameraMatrix = [fx1, 0, cx1; 0, fy1, cy1; 0, 0, 1];
rightCameraMatrix = [fx2, 0, cx2; 0, fy2, cy2; 0, 0, 1];

% Example distortion coefficients (replace with actual values)
leftDistCoeffs = [-0.1754, 0.0275, -0.0013, -0.0006, 0];
rightDistCoeffs = [-0.1711, 0.0258, -0.0010, -0.0009, 0];

% Example translation vector (replace with actual values from calibration)
T = [62.9976; -0.0024; 0.0022]; % Translation vector in meters



% Create camera parameter objects
leftCameraParams = cameraParameters('IntrinsicMatrix', leftCameraMatrix', ...
                                    'RadialDistortion', leftDistCoeffs(1:2), ...
                                    'TangentialDistortion', leftDistCoeffs(3:4), ...
                                    'ImageSize', size(leftImage(:,:,1)));

rightCameraParams = cameraParameters('IntrinsicMatrix', rightCameraMatrix', ...
                                     'RadialDistortion', rightDistCoeffs(1:2), ...
                                     'TangentialDistortion', rightDistCoeffs(3:4), ...
                                     'ImageSize', size(rightImage(:,:,1)));

% Create stereo parameters object
stereoParams = stereoParameters(leftCameraParams, rightCameraParams, R, T);

% Rectify the images
[J1, J2, reprojmatrix] = rectifyStereoImages(im1, im2, stereoParams);
I = J1;
I_smoothed = imgaussfilt(I, 2); % Adjust sigma for desired smoothing effect
im1 = I_smoothed;
I2 = J2;
I_smoothed2 = imgaussfilt(I2, 2); % Adjust sigma for desired smoothing effect
im2 = I_smoothed2;

% Load rectified images
leftImage = rgb2gray(imread('91968\Desktop\RoboticVision\project_data (1)\reference_rectifized\image_left\image_left_0.jpg'));
rightImage = rgb2gray(imread('91968\Desktop\RoboticVision\project_data (1)\reference_rectifized\image_right\image_right_0.jpg'));

% Load stereo parameters (assuming you have these parameters from stereo calibration)
%load('stereoParams.mat'); % This file should contain 'stereoParams' variable
% Detect points in the left image
pointsLeft = detectSURFFeatures(im1,'MetricThreshold', 1000);
imshow(im1);
hold on;
plot(pointsLeft);
hold off;

%[featuresLeft, validPointsLeft] = extractFeatures(I_smoothed, pointsLeft);

% Detect points in the right image
%surfDetector = vision.SURF('MetricThreshold', 100, 'NumOctaves', 3, 'NumScaleLevels', 4);
%pointsLeft = detectORBFeatures(im1, 'MetricThreshold', 2500);
[featuresLeft, validPointsLeft] = extractFeatures(im1, pointsLeft);
%surfDetector = vision.SURF('MetricThreshold', 100, 'NumOctaves', 3, 'NumScaleLevels', 4);
%pointsRight = detectSURFFeatures(im2, 'MetricThreshold', 2500);
pointsRight = detectSURFFeatures(im2,'MetricThreshold', 1000);
[featuresRight, validPointsRight] = extractFeatures(im2, pointsRight);

%[featuresRight, validPointsRight] = extractFeatures(rightImage, pointsRight);

% Match features between the images
ratioThreshold = 0.6; % Example: Increase ratio test threshold
indexPairs = matchFeatures(featuresLeft, featuresRight, 'MatchThreshold', ratioThreshold,'MatchThreshold',80);
matchedPointsLeft = validPointsLeft(indexPairs(:, 1), :);
matchedPointsRight = validPointsRight(indexPairs(:, 2), :);



% Apply RANSAC to estimate fundamental matrix and refine points
[fMatrix, inlierIdx] = estimateFundamentalMatrix(matchedPointsLeft, matchedPointsRight, 'Method', 'RANSAC', 'NumTrials', 500, 'DistanceThreshold', 200);
inlierPointsLeft = matchedPointsLeft(inlierIdx, :);
inlierPointsRight = matchedPointsRight(inlierIdx, :);
figure;
showMatchedFeatures(im1, im2, inlierPointsLeft, inlierPointsRight);
title('Refined Matched Points using RANSAC');


% Plot corresponding points
plot(inlierPointsLeft.Location(:, 1), inlierPointsLeft.Location(:, 2), 'go');

% Display epipolar lines in the right image
subplot(1, 2, 2);
imshow(im2);
title('Epipolar Lines in Right Image');
hold on;

epiLinesRight = epipolarLine(fMatrix, inlierPointsLeft.Location);
pointsRight = lineToBorderPoints(epiLinesRight, size(im2));
line(pointsRight(:, [1, 3])', pointsRight(:, [2, 4])');

% Plot corresponding points
plot(inlierPointsRight.Location(:, 1), inlierPointsRight.Location(:, 2), 'go');

hold off;

% Show the images side by side
figure;
imshowpair(im1, im2, 'montage');
title('Rectified Images Side by Side');


% Compute disparity for inlier points
disparities = inlierPointsLeft.Location(:, 1) - inlierPointsRight.Location(:, 1);

% Assume we have the camera parameters (focal length and baseline)
focalLength = 1401; % in pixels
baseline = 0.120; % in meters

% Compute the depth map
depths = (focalLength * baseline) ./ disparities;
[d,sim,dsi] = istereo(im1, im2, [400, 700],3);
b=0.120;
f= 1401;
Z =abs((b*f)./d);
imshow(d);
Z = f * baseline./d;
idisp(Z, 'square', 'new');
colormap(jet); colorbar;

% Plot the depth map for the matched points
figure;
imshow(im1); hold on;
for i = 1:length(inlierPointsLeft)
    x = inlierPointsLeft.Location(i, 1);
    y = inlierPointsLeft.Location(i, 2);
    text(x, y, sprintf('%.2f', depths(i)), 'Color', 'red');
end
idisp(depths, 'square', 'new');
colormap(jet); colorbar;
title('Depth Map for Matched Points');
[U,V] = meshgrid(1:size(im1,2),1:size(im2,1));
cx= 1150.3;
cy= 670.6;
Z = f * baseline./d;
X = ((U-cx)*Z)./f; Y = ((V-cy)*Z)./f;
Z = medfilt2(Z,[5 5]);
surf(Z);
shading interp; view(-74,44)
set(gca,ZDir="reverse"); set(gca,XDir="reverse")
colormap(parula)
