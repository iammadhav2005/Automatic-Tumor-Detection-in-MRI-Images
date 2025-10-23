tumor_folder = 'yes/';
nontumor_folder = 'no/';
tumor_imgs = dir(fullfile(tumor_folder, '*.jpg'));
nontumor_imgs = dir(fullfile(nontumor_folder, '*.jpg'));
X = [];
y = [];
for i = 1:length(tumor_imgs)
    img = imread(fullfile(tumor_folder, tumor_imgs(i).name));
    if size(img,3)==3, img = rgb2gray(img); end
    img = imresize(img, [128 128]);
    level = graythresh(img);
    bw = imbinarize(img, level);
    bw = bwareaopen(bw, 30);
    stats = regionprops(bw, 'Area');
    if isempty(stats)
        region = img > 0;
    else
        [~, bigIdx] = max([stats.Area]);
        labelBW = bwlabel(bw);
        region = ismember(labelBW, bigIdx);
    end
    region_pixels = img(region);
    f_mean = mean(region_pixels(:));
    f_std = std(double(region_pixels(:)));
    X = [X; f_mean, f_std];
    y = [y; 1];
end

for i = 1:length(nontumor_imgs)
    img = imread(fullfile(nontumor_folder, nontumor_imgs(i).name));
    if size(img,3)==3, img = rgb2gray(img); end
    img = imresize(img, [128 128]);
    level = graythresh(img);
    bw = imbinarize(img, level);
    bw = bwareaopen(bw, 30);
    stats = regionprops(bw, 'Area');
    if isempty(stats)
        region = img > 0;
    else
        [~, bigIdx] = max([stats.Area]);
        labelBW = bwlabel(bw);
        region = ismember(labelBW, bigIdx);
    end
    region_pixels = img(region);
    f_mean = mean(region_pixels(:));
    f_std = std(double(region_pixels(:)));
    X = [X; f_mean, f_std];
    y = [y; 0];
end

% ---- Train ML Classifier and Show Metrics and Bar Graph ----
metricsReady = false;
accuracy = 0; precision = 0; recall = 0; f1score = 0;
if length(y) > 4 && sum(y==1)>0 && sum(y==0)>0
    cv = cvpartition(y, 'HoldOut', 0.3);
    X_train = X(training(cv), :); y_train = y(training(cv));
    X_test = X(test(cv), :); y_test = y(test(cv));
    SVMModel = fitcsvm(X_train, y_train);
    y_pred = predict(SVMModel, X_test);
    accuracy = sum(y_pred == y_test) / numel(y_test);
    disp(['Test Accuracy: ', num2str(accuracy*100), '%']);
    cm = confusionmat(y_test, y_pred);
    disp('Confusion Matrix:');
    disp(cm);
    if numel(cm)==4
        TP = cm(2,2); FP = cm(1,2); FN = cm(2,1); TN = cm(1,1);
    else
        TP = 0; FP = 0; FN = 0; TN = 0;
    end
    precision = TP/(TP+FP+eps); % eps avoids division by 0
    recall = TP/(TP+FN+eps);
    f1score = 2*precision*recall/(precision+recall+eps);
    disp(['Precision: ', num2str(precision)]);
    disp(['Recall: ', num2str(recall)]);
    disp(['F1-score: ', num2str(f1score)]);
    metricsReady = true;

    % --- Graphical representation for ML parameters ---
    metrics_bar = [accuracy*100, precision*100, recall*100, f1score*100];
    labels = {'Accuracy','Precision','Recall','F1-score'};
    figure;
    bar(metrics_bar, 'FaceColor',[0.3,0.6,0.8]);
    set(gca,'xticklabel',labels,'FontSize',12,'FontWeight','bold');
    ylabel('Percentage (%)');
    ylim([0, 110]);
    title('ML Model Performance Metrics');
    for i=1:length(metrics_bar)
        text(i, metrics_bar(i)+3, sprintf('%.1f',metrics_bar(i)),...
            'HorizontalAlignment','center','FontSize',12,'Color','b','FontWeight','bold');
    end
else
    SVMModel = fitcsvm(X, y);
    y_pred = predict(SVMModel, X);
    accuracy = sum(y_pred == y) / numel(y);
    disp(['Training Accuracy: ', num2str(accuracy*100), '%']);
    disp('WARNING: Not enough images for test set metrics. Showing training metrics only.');
    metricsReady = false;
end

% ---- GUI Upload and 3-Panel Visualization ----
[filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'}, 'Select an MRI image for detection');
if isequal(filename,0)
    disp('User did not select an image.');
    return;
end
testImgPath = fullfile(pathname, filename);
testImg = imread(testImgPath);
if size(testImg,3)==3, testImg = rgb2gray(testImg); end
testImg = imresize(testImg, [128 128]);
level = graythresh(testImg);
bw = imbinarize(testImg, level);
bw = bwareaopen(bw, 30);
stats = regionprops(bw, 'Area', 'BoundingBox');
if isempty(stats)
    region = testImg > 0;
    box = [1 1 10 10];
else
    [~, bigIdx] = max([stats.Area]);
    labelBW = bwlabel(bw);
    region = ismember(labelBW, bigIdx);
    box = stats(bigIdx).BoundingBox;
end
region_pixels = testImg(region);
f_mean = mean(region_pixels(:));
f_std = std(double(region_pixels(:)));
features = [f_mean, f_std];
pred = predict(SVMModel, features);
tumor_mask = region;
tumor_boundary = bwperim(tumor_mask);

% --- Three-panel output
figure;
subplot(1,3,1); imshow(testImg, []); title('Brain');
subplot(1,3,2); imshow(tumor_mask); title('Tumor Alone');
subplot(1,3,3); imshow(testImg, []); title('Detected Tumor');
hold on;
[yy, xx] = find(tumor_boundary);
plot(xx, yy, 'y.', 'MarkerSize', 1.5);
if pred == 1
    text(10,10,'Tumor detected!','Color','r','FontSize',14,'FontWeight','bold');
else
    text(10,10,'No tumor detected','Color','g','FontSize',14,'FontWeight','bold');
end
hold off;

% ---- Done ----
