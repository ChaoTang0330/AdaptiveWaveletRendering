% filename = "source/cornellTrans.png_11_22_0_9.csv";
% sample1 = reshape(mean(csvread(filename,0,0,[0 0 1024*1024-1 2]),2),[1024,1024]);
% sample2 = reshape(mean(csvread(filename,0,3,[0 3 1024*1024-1 5]),2),[1024,1024]);
% sample3 = reshape(mean(csvread(filename,0,6,[0 6 1024*1024-1 8]),2),[1024,1024]);
% sample4 = reshape(mean(csvread(filename,0,9,[0 9 1024*1024-1 11]),2),[1024,1024]);

sample = cat(3,sample1,sample2,sample3,sample4);
B = transpose(mean(sample,3));
B(B>1) = 1;
sigmaB = ((max(sample,[],3) - min(sample,[],3)).^2)./((max(sample,[],3) + min(sample,[],3)+ eps).^2)./4;
sigmaB = sigmaB';

for k = 1:5
    [A1,H1,V1,D1] = wavecdf97(B,k);
    W1 = (H1.^2+ V1.^2 + D1.^2)./3;
    [cA1,cH1,cV1,cD1]=wavecdf2(sigmaB,k);
    figure(1)
    temp = cA1-W1;
    priority = (temp - min(temp,[],'all')) ./(max(temp,[],'all') - min(temp,[],'all'));
    figure(k);
    imagesc(priority);
    colorbar
    colormap(jet)
    axis equal
    axis tight
    saveas(gcf,k + ".png");
end
"finished"
% % bior4.4 9/7
% [A1,H1,V1,D1]=dwt2(B,'bior4.4');
% W1 = (H1.^2+ V1.^2 + D1.^2)./3;
% [cA1,cH1,cV1,cD1]=dwt2(sqrt(sigmaB),'bior4.4');
% cA1 = cA1.^2;
% figure(1)
% imagesc(abs(cA1-W1))
% colorbar
% colormap(jet)
% 
% [A2,H2,V2,D2]=dwt2(A1,'bior4.4');
% W2 = (H2.^2+ V2.^2 + D2.^2)./3;
% [cA2,cH2,cV2,cD2]=dwt2(sqrt(cA1),'bior4.4');
% cA2 = cA2.^2;
% figure(2)
% imagesc(abs(cA2-W2))
% colorbar
% colormap(jet)
% 
% [A3,H3,V3,D3]=dwt2(A2,'bior4.4');
% W3 = (H3.^2+ V3.^2 + D3.^2)./3;
% [cA3,cH3,cV3,cD3]=dwt2(sqrt(cA2),'bior4.4');
% cA3 = cA3.^2;
% figure(3)
% imagesc(abs(cA3-W3))
% colorbar
% colormap(jet)
% 
% [A4,H4,V4,D4]=dwt2(A3,'bior4.4');
% W4 = (H4.^2+ V4.^2 + D4.^2)./3;
% [cA4,cH4,cV4,cD4]=dwt2(sqrt(cA3),'bior4.4');
% cA4 = cA4.^2;
% figure(4)
% imagesc(abs(cA4-W4))
% colorbar
% colormap(jet)
% 
% [A5,H5,V5,D5]=dwt2(A4,'bior4.4');
% W5 = (H5.^2+ V5.^2 + D5.^2)./3;
% [cA5,cH5,cV5,cD5]=dwt2(sqrt(cA4),'bior4.4');
% cA5 = cA5.^2;
% figure(5)
% imagesc(abs(cA5-W5))
% colorbar
% colormap(jet)