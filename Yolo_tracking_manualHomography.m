% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();
frameRate = round(obj.frameRate);
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track

%YOLO stuff
cfg_file = 'cfg/yolov4.cfg';
weight_file = 'weights/yolov4.weights';
throushold = 0.8;
NMS = 0.6;

%import all classes
fid = fopen('coco.names','r');
names = textscan(fid, '%s', 'Delimiter',{'   '});
fclose(fid);classesNames = categorical(names{1});
RGB = randi(255,length(classesNames),3);

frame = readFrame(obj.reader);
frameSize = size(frame);

%homography
disp("pick corners of reference");
imshow(frame);
[aa,bb]=getpts;
P=[aa,bb]';
detail = 80;

prompt = 'Enter width in feet: ';
x = input(prompt);
w = x * detail;
prompt = 'Enter length in feet: ';
x = input(prompt);
h = x * detail;

Q = [ 0, 0, w, w; 0, h, h, 0];
H = homography(P,Q);

v = VideoWriter('output_videos/truckManualHomography', 'MPEG-4');
v.FrameRate = frameRate;
open(v);

frameCount = 0;
framesTillDetect = 0;
histLength = frameRate;

while frameCount < 400
    frame = readFrame(obj.reader);
    frameGray = rgb2gray(frame);
    if frameCount == 0
        oldFrameGray = frameGray;
    end
    frameCount = frameCount + 1;
    
    if framesTillDetect == 0
        [centroids,bboxes,annotations] = detectObjects(cfg_file,weight_file,frame,throushold,NMS,classesNames,RGB,frameSize);
        
        if ~isempty(bboxes)
            pointList = cell(length(bboxes(:,1)),1);
            for count = 1:1:length(bboxes(:,1))
                points = detectMinEigenFeatures(frameGray, 'ROI', bboxes(count, :));
                [bbox, points, centroid] = customPointTracking(frameGray,bboxes(count,:),points.Location,pointTracker,oldFrameGray,frameSize);
                pointList(count,1) = {points};
                if ~isempty(points)
                    frame = insertMarker(frame, points, '+', 'Color', 'white');
                end
            end
        else
            pointList = [];
        end
        framesTillDetect = 15;
    else
        if ~isempty(pointList) && ~isempty(bboxes)
            for count = 1:1:length(pointList(:,1))
                points = pointList{count,1};
                [bbox, points, centroid] = customPointTracking(frameGray,bboxes(count,:),points,pointTracker,oldFrameGray,frameSize);
                if ~isempty(points)
                    frame = insertMarker(frame, points, '+', 'Color', 'white');
                end
                bboxes(count,:) = bbox(1,:);
                pointList(count,1) = {points};
                centroids(count,1) = centroid(1,1);
                centroids(count,2) = centroid(1,2);
            end
        end
        framesTillDetect = framesTillDetect - 1;
    end
    
    tracks = predictNewLocationsOfTracks(tracks);
    [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment(tracks, centroids, bboxes, framesTillDetect);
    tracks = updateAssignedTracks(tracks, centroids, assignments, bboxes, annotations);
    tracks = updateUnassignedTracks(tracks, unassignedTracks);
    tracks = deleteLostTracks(tracks);
    [tracks, nextId] = createNewTracks(tracks, centroids, bboxes, annotations, nextId, unassignedDetections);
    tracks = findVelocity(tracks,H, histLength, frameRate);
    frame = displayTrackingResults(frame, tracks);
    
    oldFrameGray = frameGray;
    obj.videoPlayer.step(frame);
    writeVideo(v,frame);
end
close(v);
release(obj.videoPlayer);
release(pointTracker);
delete(obj.videoPlayer);

%tracking funtions from: https://www.mathworks.com/help/vision/ug/motion-based-multiple-object-tracking.html
function obj = setupSystemObjects()
% Create a video reader.
obj.reader = VideoReader('videos/mall_crowd.mp4');
obj.frameRate = obj.reader.FrameRate;
obj.videoPlayer = vision.VideoPlayer();
end


function tracks = initializeTracks()
% create an empty array of tracks
tracks = struct(...
    'id', {}, ...
    'class', {}, ...
    'bbox', {}, ...
    'kalmanFilter', {}, ...
    'age', {}, ...
    'totalVisibleCount', {}, ...
    'consecutiveInvisibleCount', {}, ...
    'centroidHistory', {}, ...
    'centCount', {}, ...
    'currentVelocity', {});
end


function [centroids,bboxes,annotations] = detectObjects(cfg_file,weight_file,frame,throushold,NMS,classesNames,RGB,frameSize)
outFeatures = yolov3v4Predict(cfg_file,weight_file,frame);

% Threshold filtering + NMS processing
scores = outFeatures(:,5);
outFeatures = outFeatures(scores>throushold,:);

allBBoxes = outFeatures(:,1:4);
allScores = outFeatures(:,5);
[maxScores,indxs] = max(outFeatures(:,6:end),[],2);
allScores = allScores.*maxScores;
allLabels = classesNames(indxs);

% NMS non-maximum suppression
if ~isempty(allBBoxes)
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(allBBoxes,allScores,allLabels,...
        'RatioType','Min','OverlapThreshold',NMS);
%     disp(scores);
    annotations = string(labels); % + ": " + string(scores);
    [~,ids] = ismember(labels,classesNames);
    colors = RGB(ids,:);
else
    bboxes = [];
    annotations = [];
end

% Keep bboxes inbounds for point detection
if ~isempty(bboxes)
    for count = 1:1:length(bboxes(:, 1))
        if bboxes(count,1) < 1
            bboxes(count,1) = 1;
        end
        if bboxes(count,2) < 1
            bboxes(count,2) = 1;
        end
        if bboxes(count,3) + bboxes(count,1) > frameSize(1,2)
            bboxes(count,3) = frameSize(1,2) - bboxes(count,1);
        end
        if bboxes(count,4) + bboxes(count,2) > frameSize(1,1)
            bboxes(count,4) = frameSize(1,1) - bboxes(count,2);
        end
    end
    % YOLO provided bboxes to list of centroids
    centroids(:,1) = bboxes(:,1) + 0.5 * bboxes(:,3);
    centroids(:,2) = bboxes(:,2) + 0.5 * bboxes(:,4);
else
    centroids = [];
end
end


function [bbox, visiblePoints, centroid] = customPointTracking(frameGray, bbox, points,pointTracker,oldFrameGray,frameSize)
if ~isempty(points)
    bboxPoints = bbox2points(bbox(1, :));
    initialize(pointTracker, points, oldFrameGray);
    
    [xyPoints, isFound] = step(pointTracker, frameGray);
    visiblePoints = xyPoints(isFound, :);
    oldInliers = points(isFound, :);
    
    numPts = size(visiblePoints, 1);
    
    if numPts >= 10
        % Estimate the geometric transformation between the old points and the new points.
        [xform, inlierIdx] = estimateGeometricTransform2D(oldInliers, visiblePoints, 'similarity', 'MaxDistance', 2);
        visiblePoints = visiblePoints(inlierIdx, :);
        
        % Apply the transformation to the bounding box.
        bboxPoints = transformPointsForward(xform, double(bboxPoints));
        bbox = points2bbox(bboxPoints);
    else
        points = detectMinEigenFeatures(frameGray, 'ROI', bbox);
        visiblePoints = points.Location;
    end
    
    if bbox(1,1) < 1
        bbox(1,1) = 1;
    end
    if bbox(1,2) < 1
        bbox(1,2) = 1;
    end
    if bbox(1,3) + bbox(1,1) > frameSize(1,2)
        bbox(1,3) = frameSize(1,2) - bbox(1,1);
    end
    if bbox(1,4) + bbox(1,2) > frameSize(1,1)
        bbox(1,4) = frameSize(1,1) - bbox(1,2);
    end
    
    release(pointTracker);
else
    visiblePoints = [];
end
centroid(:,1) = bbox(1,1) + 0.5 * bbox(1,3);
centroid(:,2) = bbox(1,2) + 0.5 * bbox(1,4);
end


function tracks = predictNewLocationsOfTracks(tracks)
for i = 1:length(tracks)
    bbox = tracks(i).bbox;
    
    % Predict the current location of the track.
    predictedCentroid = predict(tracks(i).kalmanFilter);
    
    tracks(i).centroidHistory(tracks(i).centCount, :) = predictedCentroid;
    tracks(i).centCount = tracks(i).centCount + 1;
    
    % Shift the bounding box so that its center is at
    % the predicted location.
    predictedCentroid = int32(predictedCentroid) - int32(bbox(3:4) / 2);
    tracks(i).bbox = [predictedCentroid, bbox(3:4)];
end
end


function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment(tracks, centroids, bboxes, framesTillDetect)

nTracks = length(tracks);
nDetections = size(centroids, 1);

% Compute the cost of assigning each detection to each track.
cost = zeros(nTracks, nDetections);
for i = 1:nTracks
    cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
end

% Solve the assignment problem.
unassignedTrackCost = 40;
unassignedDetectionCost = 40;
overlapThreshold = .3;


%found from: https://www.mathworks.com/help/vision/ug/tracking-pedestrians-from-a-moving-car.html
if framesTillDetect == 15
    if ~isempty(cost)
        overlap = zeros(length(cost(:,1)), length(cost(1,:)));
        overlapComp = ones(length(cost(:,1)), length(cost(1,:))) * overlapThreshold;
        for i = 1:nTracks
            overlap(i,:) = bboxOverlapRatio(tracks(i).bbox, bboxes, 'Min');
        end
        check = overlap < overlapComp;
        check = check * 100;
        cost = cost + check;
    end
end

[assignments, unassignedTracks, unassignedDetections] = ...
    assignDetectionsToTracks(cost, unassignedTrackCost, unassignedDetectionCost);
end


function tracks = updateAssignedTracks(tracks, centroids, assignments, bboxes, annotations)
numAssignedTracks = size(assignments, 1);
for i = 1:numAssignedTracks
    trackIdx = assignments(i, 1);
    detectionIdx = assignments(i, 2);
    centroid = centroids(detectionIdx, :);
    bbox = bboxes(detectionIdx, :);
    annotation = annotations(detectionIdx, :);
    
    % Correct the estimate of the object's location
    % using the new detection.
    correct(tracks(trackIdx).kalmanFilter, centroid);
    
    % Replace predicted bounding box with detected
    % bounding box.
    tracks(trackIdx).bbox = bbox;
    tracks(trackIdx).class = annotation;
    
    % Update track's age.
    tracks(trackIdx).age = tracks(trackIdx).age + 1;
    
    % Update visibility.
    tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;
    tracks(trackIdx).consecutiveInvisibleCount = 0;
end
end


function tracks = updateUnassignedTracks(tracks, unassignedTracks)
for i = 1:length(unassignedTracks)
    ind = unassignedTracks(i);
    tracks(ind).age = tracks(ind).age + 1;
    tracks(ind).consecutiveInvisibleCount = ...
        tracks(ind).consecutiveInvisibleCount + 1;
end
end


function tracks = deleteLostTracks(tracks)
if isempty(tracks)
    return;
end

% invisibleForTooLong = 51;
invisibleForTooLong = 2;
ageThreshold = 16;

% Compute the fraction of the track's age for which it was visible.
ages = [tracks(:).age];
totalVisibleCounts = [tracks(:).totalVisibleCount];
visibility = totalVisibleCounts ./ ages;

% Find the indices of 'lost' tracks.
lostInds = (ages < ageThreshold & visibility < 0.6) | ...
    [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

% lostInds = (visibility < 0.6) | ...
%     [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;


% Delete lost tracks.
tracks = tracks(~lostInds);
end


function [tracks, nextId] = createNewTracks(tracks, centroids, bboxes, annotations, nextId, unassignedDetections)
centroids = centroids(unassignedDetections, :);
bboxes = bboxes(unassignedDetections, :);

for i = 1:size(centroids, 1)
    
    centroid = centroids(i,:);
    bbox = bboxes(i, :);
    annotation = annotations(i, 1);
    
    % Create a Kalman filter objects
        kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
            centroid, [200, 50], [100, 25], 100);
    
    % Create a new track.
    newTrack = struct(...
        'id', nextId, ...
        'class', annotation, ...
        'bbox', bbox, ...
        'kalmanFilter', kalmanFilter, ...
        'age', 1, ...
        'totalVisibleCount', 1, ...
        'consecutiveInvisibleCount', 0, ...
        'centroidHistory', centroid, ...
        'centCount', 1, ...
        'currentVelocity', '');
    
    % Add it to the array of tracks.
    tracks(end + 1) = newTrack;
    
    % Increment the next id.
    nextId = nextId + 1;
end


end


function tracks = findVelocity(tracks, H, histLength, frameRate)
for i = 1:length(tracks)
if tracks(i).centCount == histLength
    homoPointsP = e2h(tracks(i).centroidHistory');
    homoPointsQ = H * homoPointsP;
    pointsQ = h2e(homoPointsQ);
    pointsQ = pointsQ';
    
    dif = zeros((length(pointsQ)-1), 2);
    for c = 1:(length(pointsQ)-1)
        dif(c,1) = pointsQ((c+1),1) - pointsQ((c),1);
        dif(c,1) = dif(c,1) / 80;
        dif(c,2) = pointsQ((c+1),2) - pointsQ((c),2);
        dif(c,2) = dif(c,2) / 80;
    end
    
    dif2 = sqrt(dif(:,1).^2 + dif(:,2).^2);
%     dif2 = smoothdata(dif2);
    dif2 = dif2 * frameRate;
    dif3 = sum(dif2) / length(dif2);
    dif3 = dif3 * 0.681818;
    tracks(i).currentVelocity = dif3;
    disp("Object ID " + tracks(i).id + ": " + dif3 + " MPH");
    tracks(i).centCount = 1;
end
end
end


function frame = displayTrackingResults(frame, tracks)
% Convert the frame and the mask to uint8 RGB.
frame = im2uint8(frame);

minVisibleCount = 8;
if ~isempty(tracks)
    
    % Noisy detections tend to result in short-lived tracks.
    % Only display tracks that have been visible for more than
    % a minimum number of frames.
    reliableTrackInds = [tracks(:).totalVisibleCount] > minVisibleCount;
    reliableTracks = tracks(reliableTrackInds);
    
    % Display the objects. If an object has not been detected
    % in this frame, display its predicted bounding box.
    if ~isempty(reliableTracks)
        % Get bounding boxes.
        bboxes = cat(1, reliableTracks.bbox);
        
        % Get ids.
        ids = int32([reliableTracks(:).id]);
        for count = 1:1:length(reliableTracks)
            classes(count, 1) = reliableTracks(count).class;
            if ~isempty(reliableTracks(count).currentVelocity)
                currentVelocity(count, 1) = reliableTracks(count).currentVelocity;
            end
        end
        
        % Create labels for objects indicating the ones for
        % which we display the predicted rather than the actual
        % location.
        
        for count = 1:1:length(reliableTracks)
            if ~isempty(reliableTracks(count).currentVelocity)
                labels(count,1) = string(classes(count, 1)) + "   ID: " + int2str(ids(1, count)') + " Speed: " + currentVelocity(count, :) + " MPH";
            else
                labels(count,1) = string(classes(count, 1)) + "   ID: " + int2str(ids(1, count)');
            end
        end

%         labels = string(classes) + "   ID: " + int2str(ids') + " Speed: " + currentVelocity + " MPH";
        predictedTrackInds = [reliableTracks(:).consecutiveInvisibleCount] > 0;
        isPredicted = cell(size(labels));
        isPredicted(predictedTrackInds) = {' predicted'};
        labels = strcat(labels, isPredicted);
        
        % Draw the objects on the frame.
        frame = insertObjectAnnotation(frame, 'rectangle', bboxes, labels);
    end
end
end