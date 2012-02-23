% Generate samples from the model
colormap(gray(64));

load mnist_weights;
load params;
% numhid=1000; numpen=500; numpen2=250; numopen=30;

fourth_rec = w4(1:size(w4,1) - 1, :);
fourth_rec_bias = w4(size(w4,1),:);

fourth_gen = w5(1:size(w5,1) - 1, :);
fourth_gen_bias = w5(size(w5,1),:);

third = w6(1:size(w6,1) - 1, :);
third_bias = w6(size(w6,1),:);

second = w7(1:size(w7,1) - 1, :);
second_bias = w7(size(w7,1),:);

first = w8(1:size(w8,1) - 1, :);
first_bias = w8(size(w8,1),:);

numsamples = 1; % how many sample images to generate
numdims = 784;

for s = 1:numsamples

    fprintf(1, 'Generating sample %3i... \n', s);

    % Start from random final units
    fourth_states = rand(numopen, 1)';

    % Run Gibbs sampling for many iterations
    for i = 1:200
        third_states = sigmoid(fourth_states*fourth_gen + fourth_gen_bias);
        third_states = third_states > rand(size(third_states));

        fourth_states = third_states*fourth_rec + fourth_rec_bias;

        if mod(i, 20) == 0
            % Go back down
            third_states = sigmoid(fourth_states*fourth_gen + fourth_gen_bias);
            third_states = third_states > rand(size(third_states));

            second_states = sigmoid(third_states*third + third_bias);
            second_states = second_states > rand(size(second_states));

            first_states = sigmoid(second_states*second + second_bias);
            first_states = first_states > rand(size(first_states));

            visible = sigmoid(first_states*first + first_bias);

            % Convert the visible states into an image
            pixels = uint8(round(visible .* 64));
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
