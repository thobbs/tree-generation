% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200.

maxepoch=200;
fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');

load mnistvh
load mnisthp
load mnisthp2
load mnistpo

makebatches;
[numcases numdims numbatches] = size(batchdata);
N=numcases;

% Preinitialize weights of the autoencoder
w1 = [vishid;   hidrecbiases];
w2 = [hidpen;   penrecbiases];
w3 = [hidpen2;  penrecbiases2];
w4 = [hidtop;   toprecbiases];

w5 = [hidtop';  topgenbiases];
w6 = [hidpen2'; hidgenbiases2];
w7 = [hidpen';  hidgenbiases];
w8 = [vishid';  visbiases];


l1 = size(w1,1) - 1;
l2 = size(w2,1) - 1;
l3 = size(w3,1) - 1;
l4 = size(w4,1) - 1;
l5 = size(w5,1) - 1;
l6 = size(w6,1) - 1;
l7 = size(w7,1) - 1;
l8 = size(w8,1) - 1;
l9 = l1;
test_err  = [];
train_err = [];


for epoch = 1:maxepoch

    % Compute training reconstruction error
    err=0;
    [numcases numdims numbatches] = size(batchdata);
    N = numcases;
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)];
        data = [data ones(N,1)];

        w1probs = sigmoid(data*w1);     w1probs = [w1probs ones(N,1)];
        w2probs = sigmoid(w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = sigmoid(w2probs*w3)); w3probs = [w3probs ones(N,1)];
        w4probs = w3probs*w4;           w4probs = [w4probs ones(N,1)];
        w5probs = sigmoid(w4probs*w5)); w5probs = [w5probs ones(N,1)];
        w6probs = sigmoid(w5probs*w6)); w6probs = [w6probs ones(N,1)];
        w7probs = sigmoid(w6probs*w7)); w7probs = [w7probs ones(N,1)];
        dataout = sigmoid(w7probs*w8));

        err += (1/N) * sum(sum( (data(:,1:end-1)-dataout).^2 ));
    end
    train_err(epoch)=err/numbatches;


    % Display figures to compare real data and reconstructions
    fprintf(1,'Top row, real data; Bottom row, reconstructions\n');
    output=[];
    for ii=1:15
        output = [output data(ii,1:end-1)' dataout(ii,:)'];
    end

    if epoch==1
        close all
        figure('Position', [100,600,1000,200]);
    else
        figure(1)
    end
    mnist_display(output);
    drawnow;


    % Compute test reconstruction error
    [testnumcases testnumdims testnumbatches] = size(testbatchdata);
    N = testnumcases;
    err=0;
    for batch = 1:testnumbatches
        data = [testbatchdata(:,:,batch)];
        data = [data ones(N,1)];

        w1probs = sigmoid(data*w1));    w1probs = [w1probs ones(N,1)];
        w2probs = sigmoid(w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = sigmoid(w2probs*w3)); w3probs = [w3probs ones(N,1)];
        w4probs = w3probs*w4;           w4probs = [w4probs ones(N,1)];
        w5probs = sigmoid(w4probs*w5)); w5probs = [w5probs ones(N,1)];
        w6probs = sigmoid(w5probs*w6)); w6probs = [w6probs ones(N,1)];
        w7probs = sigmoid(w6probs*w7)); w7probs = [w7probs ones(N,1)];
        dataout = sigmoid(w7probs*w8));

        err += (1/N) * sum(sum( (data(:,1:end-1)-dataout).^2 ));
    end
    test_err(epoch) = err/testnumbatches;
    fprintf(1,'Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \n', ...
            epoch, train_err(epoch), test_err(epoch));


    big_batch_num = 0;
    for batch = 1:numbatches/10

        fprintf(1,'epoch %d batch %d\r',epoch,batch);

        % Combine 10 minibatches into 1 larger minibatch
        data=[];
        for small_batch_num = 1:10
           data=[data batchdata(:,:, big_batch_num*10 + small_batch_num)];
        end
        big_batch_num += 1;

        % Perform conjugate gradient with 3 linesearches
        max_iter = 3;
        VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
        Dim = [l1; l2; l3; l4; l5; l6; l7; l8; l9];

        [X, fX] = minimize(VV, 'cg_mnist', max_iter, Dim, data);

        w1 = reshape(X(1:(l1+1)*l2), l1+1, l2);
        xxx  = (l1+1) * l2;
        w2 = reshape(X(xxx+1:xxx+(l2+1)*l3), l2+1, l3);
        xxx += (l2+1) * l3;
        w3 = reshape(X(xxx+1:xxx+(l3+1)*l4), l3+1, l4);
        xxx += (l3+1) * l4;
        w4 = reshape(X(xxx+1:xxx+(l4+1)*l5), l4+1, l5);
        xxx += (l4+1) * l5;
        w5 = reshape(X(xxx+1:xxx+(l5+1)*l6), l5+1, l6);
        xxx += (l5+1) * l6;
        w6 = reshape(X(xxx+1:xxx+(l6+1)*l7), l6+1, l7);
        xxx += (l6+1) * l7;
        w7 = reshape(X(xxx+1:xxx+(l7+1)*l8), l7+1, l8);
        xxx += (l7+1) * l8;
        w8 = reshape(X(xxx+1:xxx+(l8+1)*l9), l8+1, l9);

    end

    save mnist_weights w1 w2 w3 w4 w5 w6 w7 w8
    save mnist_error test_err train_err;

end

