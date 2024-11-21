close all;
clear all;

im1 = imread("91968\Desktop\RoboticVision\project_data (1)\image_left_unrect\image_left_unrect_11.jpg");
im2 = imread("91968\Desktop\RoboticVision\project_data (1)\image_right_unrect\image_right_unrect_11.jpg");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RECTIFICATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert the rotation vector to a rotation matrix
rx = 0.0034;
ry = 0.0000;
rz = -0.0008;
rotationVector = [rx, ry, rz];
R = rotationVectorToMatrix( rotationVector ); 
% Example translation vector (replace with actual values from calibration)
T = [62.9976; -0.0024; 0.0022]; % Translation vector in meters

% Example intrinsic parameters (replace with actual values from calibration)
fx1 = 1401.1400; fy1 = 1401.1400; cx1 = 1150.3900; cy1 = 670.6210; % Intrinsic parameters for left camera
fx2 = 1399.2200; fy2 = 1399.2200; cx2 = 1115.0000; cy2 = 600.0290; % Intrinsic parameters for right camera

leftCameraMatrix = [fx1, 0, cx1; 0, fy1, cy1; 0, 0, 1];
rightCameraMatrix = [fx2, 0, cx2; 0, fy2, cy2; 0, 0, 1];

% Example distortion coefficients (replace with actual values)
leftDistCoeffs = [-0.1754, 0.0275, -0.0013, -0.0006, 0];
rightDistCoeffs = [-0.1711, 0.0258, -0.0010, -0.0009, 0];


% Create camera parameter objects
leftCameraParams = cameraParameters('IntrinsicMatrix', leftCameraMatrix', ...
                                    'RadialDistortion', leftDistCoeffs(1:2), ...
                                    'TangentialDistortion', leftDistCoeffs(3:4), ...
                                    'ImageSize', size(im1(:,:,1)));

rightCameraParams = cameraParameters('IntrinsicMatrix', rightCameraMatrix', ...
                                     'RadialDistortion', rightDistCoeffs(1:2), ...
                                     'TangentialDistortion', rightDistCoeffs(3:4), ...
                                     'ImageSize', size(im2(:,:,1)));

% Create stereo parameters object
stereoParams = stereoParameters(leftCameraParams, rightCameraParams, R, T);

% Rectify the images
[J1, J2, reprojmatrix] = rectifyStereoImages(im1, im2, stereoParams);

%%%%%%%%%%%%%%%%%%%%%%%%DISPARITY%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the depth map
[d,sim,dsi] = istereo(J1, J2, [300, 700],3);
%%%%%%%%%%%%%%%%%%%%%%%%POINT CLOUD%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xyzPoints = reconstructScene(d,reprojmatrix);
xyzPoints = xyzPoints ./ 1000;%values in meters
% Compute point cloud
ptCloud = pointCloud(xyzPoints,Color=J1);
figure
pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
%%%%%%%%%%%%%%%%%%%%%%%%DEPTH%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Camera parameters
b=0.120;
f= 1401;
u0 = 1150.39;
v0 = 670;
Z =abs((b*f)./d);
imshow(d);
Z = f * b./d;
%Depth map
idisp(Z, 'square', 'new');
colormap(jet); colorbar;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%OBJECT COLOUR IDENTIFICATION%%%%%%%%%%%%%%%%%%%
%Blue tape identification
[bluetape_picture_BW,y] = detectblue2(J1);
I = bwareaopen(bluetape_picture_BW,100);%Remove stray pixels from being considered
figure()
h = imshow(bluetape_picture_BW);
axis on
stats = regionprops('table',I,'Centroid','MajorAxisLength','Orientation');
nObjects = size(stats,1);
head(stats);
hold on
ph1 = plot(stats.Centroid(:,1),stats.Centroid(:,2),'rs');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%WORLD POINT CALCULATION%%%%%%%%%%%%%%%%%%%%%%%%
%3D Point calculation from 2D Values and camera parameters for point1
disp1 = d(round(stats.Centroid(1,2)),round(stats.Centroid(1,1)));
Z1= abs((b * f)./disp1);
X1 = ((stats.Centroid(1,1) - u0)*Z1)./f;
Y1 = ((stats.Centroid(1,2) - v0)*Z1)./f;
point1 = [X1; Y1; Z1];
%3D Point calculation from 2D Values and camera parameters for point2
disp2 = d(round(stats.Centroid(2,2)),round(stats.Centroid(2,1)));
Z2= abs((b * f)./disp2);
X2 = ((stats.Centroid(1,1) - u0)*Z2)./f;
Y2 = ((stats.Centroid(1,2) - v0)*Z2)./f;
point2 = [X2; Y2; Z2];
directionVector = [X1-X2; Y1-Y2; Z1-Z2];
unitVect = directionVector./norm(directionVector);
newpoint = point2 + (0.85* unitVect);%0.85 is mid point of second tape till the tip
%%%%%%%%TRANSFORMATION TO NEW COORDINATES FROM CAMERA1 COORDINATES%%%%%%%%%
%3D Point calculation from 2D Values and camera parameters for point1
[yellowmarker_picture_BW,y] = detectyellow(J1);
I = bwareaopen(yellowmarker_picture_BW,100); 
figure()
h = imshow(yellowmarker_picture_BW);
axis on
stats = regionprops('table',I,'Centroid','MajorAxisLength','Orientation');
nObjects = size(stats,1);
head(stats);
hold on
ph_yel = plot(stats.Centroid(:,1),stats.Centroid(:,2),'rs');
disp_yel = d(round(stats.Centroid(1,2)),round(stats.Centroid(1,1)));
Z_yel= abs((b * f)./disp_yel);
X_yel = ((stats.Centroid(1,1) - u0)*Z_yel)./f;
Y_yel = ((stats.Centroid(1,2) - v0)*Z_yel)./f;
R = eye(3); % Identity rotation matrix (no rotation)
t = [X_yel; Y_yel; Z_yel]; % Translation vector

% Constructing homogeneous transformation matrix manually
T = eye(4); % Initialize a 4x4 identity matrix
T(1:3, 1:3) = R; % Assign rotation matrix R
T(1:3, 4) = t; % Assign translation vector t

%%%%%%%%%%TIP IDENTIFICATION USING COLOR SEGMENTATION%%%%%%%%%%%%%%%%%%%%%%
%Since tip identification from the blue colour tape is not precise,
%identification is done directly by colour segmentation of tip
[tip_picture_BW,y] = tipidentification(J1);
I = bwareaopen(tip_picture_BW,100); 
BW = I; % Replace with the actual path to your BW image

% Calculate the properties of the segmented region
stats = regionprops(BW, 'Centroid', 'PixelList');

% Assume there is only one region, get the centroid and pixel list
centroid = stats.Centroid;
pixelList = stats.PixelList;

% Find the pixel with the smallest x-coordinate
%From analysing the scene, the needle rod is always such that the
%x(image)coordinate of pixel has the least x value
[~, minIdx] = min(pixelList(:,1));
least_x_pixel = pixelList(minIdx, :);

%3D re-construction of the tip
disp_Tip = d(round(least_x_pixel(2)),round(least_x_pixel(1)));   
Z_tip= abs((b * f)./disp_Tip);
X_tip = ((least_x_pixel(1) - u0)*Z_tip)./f;
Y_tip = ((least_x_pixel(2) - v0)*Z_tip)./f;
newpoint = [X_tip; Y_tip; Z_tip];
% Transform P to the new frame using T
P_new_homogeneous = T * [newpoint; 1]; % Homogeneous coordinates transformation

% Extract the transformed coordinates
P_new = P_new_homogeneous(1:3);
imgWithMarker = insertMarker(J1, [least_x_pixel(1), least_x_pixel(2)], 'x', 'Color', 'red', 'Size', 10);
imgWithText = insertText(imgWithMarker, [least_x_pixel(1), least_x_pixel(2)], sprintf('X: %.2f, Y: %.2f, Z: %.2f', P_new(1), P_new(2), P_new(3)), 'FontSize', 12, 'TextColor', 'white', 'BoxOpacity', 0);
imshow(imgWithText);
imwrite(imgWithText,"91968\Desktop\RoboticVision\project_data (1)\new\Image_with_markers.png");
hold off;


