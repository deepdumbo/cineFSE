clear all;
close all;

load('../cube_3D_original.mat')
load('../cube_3D_recon.mat')
load('../cube_3D_backward.mat')
% difference = recon - original;
% figure(1);

% subplot(2,2,1)
% imshow(squeeze(original(224,:,:)),[0, 64])
% xlabel('z');
% ylabel('y');
% title('full-sampled')
% subplot(2,2,2)
% imshow(squeeze(backward(224,:,:)),[0,64])
% title('21.4% subsampled')
% subplot(2,2,3)
% imshow(squeeze(recon(224,:,:)),[0,64])
% title('CS recon')
% subplot(2,2,4)
% imshow(squeeze(recon(224,:,:)-original(224,:,:)),[0,64])
% title('error')
max_number = max(abs(recon(:)));
scale_number = max_number/10;

original = original/scale_number;
backward = backward/scale_number;
recon = recon/scale_number;

% title('showing y-z dimension')
for sli=1:80
    sli 
    figure(2);

    subplot(2,2,1)
    imshow(squeeze(original(:,:,sli)) )
    xlabel('y');
    ylabel('x');
    title('full-sampled')
    subplot(2,2,2)
    imshow(squeeze(backward(:,:,sli)) )
    title('41% subsampled')
    subplot(2,2,3)
    imshow(squeeze(recon(:,:,sli)) )
    title('CS recon')
    subplot(2,2,4)
    imshow(squeeze(recon(:,:,sli)-original(:,:,sli)),[])
    title('error')
    pause
end