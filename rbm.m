% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning
colormap(gray(64));

epsilonw  = 0.1;   % Learning rate for weights
epsilonvb = 0.1;   % Learning rate for biases of visible units
epsilonhb = 0.1;   % Learning rate for biases of hidden units

weightcost = 0.0002;

initialmomentum = 0.5;
finalmomentum   = 0.9;

[numcases numdims numbatches] = size(batchdata);

if restart == 1
  restart=0;
  epoch=1;

  % Symmetric weights
  vishid = 0.1 * randn(numdims, numhid);

  % Biases
  hidbiases = zeros(1, numhid);
  visbiases = zeros(1, numdims);

  poshidprobs = zeros(numcases, numhid);
  neghidprobs = zeros(numcases, numhid);

  posprods    = zeros(numdims, numhid);
  negprods    = zeros(numdims, numhid);

  % For tracking momentum
  vishidinc  = zeros(numdims, numhid);
  hidbiasinc = zeros(1, numhid);
  visbiasinc = zeros(1, numdims);

  batchposhidprobs=zeros(numcases, numhid, numbatches);
end

for epoch = epoch:maxepoch

    fprintf(1, 'epoch %d\r', epoch);
    errsum=0;

    for batch = 1:20
    %for batch = 1:numbatches
        fprintf(1, 'epoch %d batch %d\r', epoch, batch);

        %%% Start positive phase %%%
        data = batchdata(:,:,batch);
        poshidprobs = sigmoid(data*vishid + repmat(hidbiases, numcases, 1));
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods = data' * poshidprobs;

        poshidact = sum(poshidprobs);
        posvisact = sum(data);

        if mod(batch, 100) == 0 && mod(batch, 200) != 0
            clf;
            pixels = [poshidprobs];
            pixels = [pixels; repmat(ones(1, numhid), 5, 1)];
            pixels = [pixels; repmat(sigmoid(hidbiases), 10, 1)];
            pixels = [pixels; repmat(ones(1, numhid), 5, 1)];

            vis1 = visbiases(1:length(visbiases)/2);
            vis1 = repmat([sigmoid(vis1) ones(1, numhid - length(vis1))], 10, 1);

            vis2 = visbiases((length(visbiases)/2)+1:length(visbiases));
            vis2 = repmat([sigmoid(vis2) ones(1, numhid - length(vis2))], 10, 1);

            pixels = [pixels; vis1];
            pixels = [pixels; vis2];
            pixels = uint8(round(pixels .* 64));
            imagesc(pixels);

            axis image off
            drawnow;
        end

        if numdims == (28*28) && mod(batch, 200) == 0
            rows = [];
            row = [];
            for i = 1:numhid;
                w_i = vishid(:,i);
                w_i = sigmoid(w_i);
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
        end

        %%% End of positive phase %%%
        poshidstates = poshidprobs > rand(numcases, numhid);

        %%% Start negative phase %%%
        negdata = sigmoid(poshidstates*vishid' + repmat(visbiases, numcases, 1));
        neghidprobs = sigmoid(negdata*vishid + repmat(hidbiases, numcases, 1));
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata);

        %%% End of negative phase %%%
        err = sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;

        if epoch > 5
          momentum = finalmomentum;
        else
          momentum = initialmomentum;
        end

        %%% Update weights and biases %%%
        vishidinc = momentum*vishidinc + ...
                    epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
     end

     fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);

     if numdims == (28*28)
         rows = [];
         row = [];
         for i = 1:numhid;
             w_i = vishid(:,i);
             w_i = sigmoid(w_i + visbiases');
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
     end

end;
