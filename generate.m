% Generate samples from the model

load mnistvhclassify;
load mnisthpclassify;
load mnisthp2classify;
% load params;
numhid=500; numpen=500; numpen2=2000;

% final hidden:  hidpen2 penrecbiases2 hidgenbiases2
% second hidden: hidpen  penrecbiases  hidgenbiases
% first hidden:  vishid  hidrecbiases  visbiases

rows = [];
row = [];
numsamples = 10; % how many sample images to generate

for s = 1:numsamples

    fprintf(1, 'Generating sample %3i... \n', s);

    % Start from random visible units
    penstates = rand(numpen, 1)';

    % Run Gibbs sampling for 500 iterations
    for i = 1:500
        % Set the hidden units
        pen2probs = sigmoid(penstates*hidpen2 + penrecbiases2);
        pen2states = pen2probs > rand(1, numpen2);

        % Generate new visible units from the model
        penstates = sigmoid(pen2states*hidpen2' + hidgenbiases2);
    end

    % Go down to the second layer
    penstates = penstates > rand(1, numpen);
    hidstates = sigmoid(penstates*hidpen + hidgenbiases);
    hidstates = rand(1, numhid);

    % Generate the visible layer
    visstates = sigmoid(hidstates*vishid' + visbiases);

    % Convert the visible states into an image
    pixels = uint8(round(visstates .* 64));
    pixels = reshape(pixels, 28, 28)';
    row = [row, pixels];
    if mod(i, 10) == 0
        rows = [rows; row];
        row = [];
    end

    if length(row) > 0
        for i = 1:(10 - length(row))
            row = [row, zeros(28, 28)];
        end
        rows = [rows; row];
    end
end

save gensamples rows;

colormap(gray(64));
imagesc(rows);
axis image off
drawnow;
pause;
