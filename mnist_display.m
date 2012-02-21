function [err] = mnist_display(digits);
% display a group of MNIST images
col=28;
row=28;

[dd,N] = size(digits);
imdisp = zeros(2*28, ceil(N/2)*28);

for nn=1:N
    ii=rem(nn,2);
    if(ii==0)
        ii=2;
    end
    jj=ceil(nn/2);

    img1 = reshape(digits(:,nn), row, col);
    img2(((ii-1)*row+1):(ii*row),((jj-1)*col+1):(jj*col)) = img1';
end

imagesc(img2,[0 1]); colormap gray; axis equal; axis off;
drawnow;
err = 0;
