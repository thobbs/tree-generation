% Generate samples from the model
colormap(gray(64));

load mnistvhclassify;
load mnisthpclassify;
load mnisthp2classify;
% load params;
numhid=500; numpen=500; numpen2=2000;

% final hidden:  hidpen2 penrecbiases2 hidgenbiases2
% second hidden: hidpen  penrecbiases  hidgenbiases
% first hidden:  vishid  hidrecbiases  visbiases

size(hidpen2)
size(hidpen)
size(vishid)

rows = [];
row = [];
for i = 1:numpen;
    w_i = (hidpen(:,i)') * vishid';
    w_i = sigmoid(w_i + visbiases);
    w_i = reshape(w_i, 28, 28)';
    row = [row, w_i];
    if mod(i, 30) == 0
        rows = [rows; row];
        row = [];
    end
end

if length(row) > 0
    numentries = length(row)/28;
    for i = 1:(30 - numentries)
        row = [row, zeros(28, 28)];
    end
    rows = [rows; row];
end

imshow(rows, [0.0 1.0]);
axis image off
drawnow;
pause;

size(hidgenbiases2)
size(hidgenbiases)

rows = [];
row = [];
for i = 1:numpen2;
    foos = hidpen2(:,i)' * hidpen';
    foos = sigmoid(foos);
    w_i = foos * vishid';
    w_i = sigmoid(w_i + visbiases);
    w_i = reshape(w_i, 28, 28)';
    row = [row, w_i];
    if mod(i, 60) == 0
        rows = [rows; row];
        row = [];
    end
end

if length(row) > 0
    numentries = length(row)/28;
    for i = 1:(60 - numentries)
        row = [row, zeros(28, 28)];
    end
    rows = [rows; row];
end

imshow(rows, [0.0 1.0]);
axis image off
drawnow;
pause;

numsamples = 1; % how many sample images to generate
numdims = 784;

for s = 1:numsamples

    fprintf(1, 'Generating sample %3i... \n', s);

    % Start from random final units
    pen2states = rand(numpen2, 1)' - 0.5;
    pen2states = pen2states > rand(size(pen2states));

    % Run Gibbs sampling for 500 iterations
    for i = 1:500
        penstates = sigmoid(pen2states*hidpen2' + hidgenbiases2);

        pen2states = sigmoid(penstates*hidpen2 + penrecbiases2);
        pen2states = pen2states > rand(size(pen2states));

        fprintf(1, 'pen2states: %d\n', length(pen2states(pen2states == 1)));
        penstates = penstates > rand(size(penstates));
        fprintf(1, 'penstates: %d\n', length(penstates(penstates == 1)));

        if mod(i, 100) == 0
            % Go back down
            hidstates = sigmoid(penstates*hidpen + hidgenbiases);
            %hidstates = hidstates > rand(size(hidstates));
            %fprintf(1, 'hidstates: %d\n', length(hidstates(hidstates == 1)));

            visstates = sigmoid(hidstates*vishid' + visbiases);
            %visstates = visstates > rand(size(visstates));
            %fprintf(1, 'visstates: %d\n', length(visstates(visstates == 1)));

            % Convert the visible states into an image
            pixels = uint8(round(visstates .* 64));
            pixels = reshape(pixels, 28, 28)';

            clf;
            imagesc(pixels);
            axis image off
            drawnow;
            sleep(0.50);
        end
    end
    pause;

end
