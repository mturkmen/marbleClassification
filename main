file={'G1';'G2';'G3'}; for j=1:3
konum=['D:\matlab\bin\images\',file{j,1},'\']; for i=1:10:41
a=imread([konum,num2str(i),'.jpg']); [s1,s2,s3]=size(a);
%% imshow(a);
a=a(:,220:3500,:);
esik=graythresh(a); % esik degeri bul abw=im2bw(a,esik); % binay convert to image
bw_erode=imerode(abw,ones(4)); % morfolojik islem bw_dilate=imdilate(bw_erode,ones(4)); % morfolojik islem bw_ar=bwareaopen(bw_dilate,100);% white objects smaller than 100
destroy
bw_hol=imfill(bw_ar,'holes'); % closing small holes
bwlbl=bwlabel(bw_hol,8); % label objects in the picture
st=regionprops(bwlbl,'BoundingBox');% coordinates of the rectangle surrounding the objects in the tagged image
xsol=floor(st.BoundingBox(1)); yust=floor(st.BoundingBox(2)); xsag=floor(xsol+st.BoundingBox(3)); yalt=floor(yust+st.BoundingBox(4)); mermer=a(yust:yalt,xsol:xsag,:);
figure(1);
subplot(2,3,1);
imshow(mermer);
title('Orjinal');
%% title(num2str(i));
%%Logarithmic Conversion
log=mermer;
imgLog = log10(1+256*im2double(log));
imgLog =(imgLog ‐ min(imgLog(:))) ./ max(imgLog(:) ‐ min(imgLog(:)));
subplot(2,3,2); imshow(imgLog);
title(' Logarithmic Conversion');
%%Gamma Verification
gama=mermer;

imgG06 = double(gama).^(0.6); imgG04 = double(gama).^(0.4); imgG03 = double(gama).^(0.3);
imgG06 = (imgG06 ‐ min(imgG06(:))) ./ max(imgG06(:) ‐ min(imgG06(:))); imgG04 = (imgG04 ‐ min(imgG04(:))) ./ max(imgG04(:) ‐ min(imgG04(:))); imgG03 = (imgG03 ‐ min(imgG03(:))) ./ max(imgG03(:) ‐ min(imgG03(:))); subplot(2,3,3);
imshow(imgG04); title('Gamma Verification ');
%%Discrete Cosine Transform (DCT)
I = rgb2gray(mermer); J = dct2(I);
J(abs(J) < 10) = 0;
K = idct2(J); subplot(2,3,4); imshow(K,[0 255]) ; title('Discrete Cosine Transform (DCT)');
%%Gabor Transformation
lambda = 8;
theta = 0;
psi = [0 pi/2];
gamma = 0.5;
bw = 1;
N =8;
img_in = im2double(marble);
img_in(:,:,2:3) = []; img_out = zeros(size(img_in,1), size(img_in,2), N); for n=1:N
gb = gabor_fn(bw,gamma,psi(1),lambda,theta)+ 1i * gabor_fn(bw,gamma,psi(2),lambda,theta);
img_out(:,:,n) = imfilter(img_in, gb, 'symmetric');
theta = theta + 2*pi/N;
end
img_out_disp = sum(abs(img_out).^2, 3).^0.5; img_out_disp = img_out_disp./max(img_out_disp(:)); subplot(2,3,5);
imshow(img_out_disp);
title('gabor output');
%% Wiener Transform
grayImage = rgb2gray(mermer);
afferNoise = imnoise(grayImage,'gaussian',0,0.025); afterWiener = wiener2(afferNoise,[6 6]); subplot(2,3,6);

imshow(afterWiener); title('Wiener Transform');
end end

%Based on a certain interval for the dark blue part of the image (variable b) 
%Graphs belong to the cropped section.
The Graph of Change of Image Brightness for %R,G,B Environments is figured last.
clear all;
close all;
sayac=0; file={'G1';'G2';'G3'};
sayac=0; for j=1:3
konum=['D:\matlab\bin\images\',file{j,1},'\']; for i=1:1:41
a=imread([konum,num2str(i),'.jpg']); aa=a(:,3757:3893,:);
imshow(aa);
title([num2str(j),' _ ',num2str(i)]); pause(0.1);
sayac=sayac+1; redv(sayac)=mean(mean(aa(:,:,1))); grenv(sayac)=mean(mean(aa(:,:,2))); bluev(sayac)=mean(mean(aa(:,:,3)));
end end
figure
plot(redv,'r')
hold on
plot(grenv,'g')
plot(bluev,'b')
legend('Red','Green','Blue');
title(' "mavi_parca" R,G,B Ortamları İçin Görüntü Parlaklıklarının Değişim Grafiği');

load select.mat vv_first vv_last fismat_first fismat_last covar_first covar_last mean_last mean_first;
class={'D';'D1D';'D1A';'M';'S'};
alan=[];
feature=[];
file=[dname, filename];
I = imread(file); I2=im2bw(I,0.2); I2=medfilt2(I2, [10 10]); [L,LL]=bwlabel(I2);
for ii=1:max(LL) alan(ii)=sum(sum(L==ii));
end
[t,ind]=max(alan);
[v,vv]=find(L==ind);
clear alan;
I3=I(min(v):max(v),min(vv):max(vv),:); index=rgb2gray(I3)>50; I3=I(min(v)+15:max(v)‐15,min(vv)+15:max(vv)‐15,:); imshow(I3);
%%% farklı renk uzayları %%%%%%%%%%%%%
Im{1,1,1}=rgb2ntsc(I3); Im{1,2,1}=colorspace('YUV<‐RGB',I3); Im{1,3,1}=double(rgb2cmy(I3)); ff=[];
feat=[]; for k=1:3
feature=[ff];
if feature(:,3)>‐0.025; [pos1,val1]=bayesian_test(feature(:,[9,16]),covar_ilk,orta_ilk,2);
drm=(class{round(pos1),1});
set(handles.edit7,'String',class{round(pos1),1}); else
[pos1,val1]=bayesian_test(feature(:,[9,3]),covar_last,mean_last,3); drm=(class{round(pos1)+2,1}); set(handles.edit7,'String',class{round(pos1)+2,1})
end
set(handles.edit8,'String',num2str(say));
function Image = hsv(Image,SrcSpace)
% Convert to HSV
Image = rgb(Image,SrcSpace);
V = max(Image,[],3);
S = (V - min(Image,[],3))./(V + (V == 0));
Image(:,:,1) = rgbtohue(Image);
Image(:,:,2) = S;
Image(:,:,3) = V;
Return;
function Image = hsl(Image,SrcSpace)
% Convert to HSL
switch SrcSpace
case 'hsv'
   % Convert HSV to HSL
   MaxVal = Image(:,:,3);
   MinVal = (1 - Image(:,:,2)).*MaxVal;
   L = 0.5*(MaxVal + MinVal);
   temp = min(L,1-L);
   Image(:,:,2) = 0.5*(MaxVal - MinVal)./(temp + (temp ==
0));
   Image(:,:,3) = L;
otherwise
   Image = rgb(Image,SrcSpace);  % Convert to sRGB
   % Convert sRGB to HSL
   MinVal = min(Image,[],3);
   MaxVal = max(Image,[],3);
   L = 0.5*(MaxVal + MinVal);
   temp = min(L,1-L);
   S = 0.5*(MaxVal - MinVal)./(temp + (temp == 0));
   Image(:,:,1) = rgbtohue(Image);
   Image(:,:,2) = S;
   Image(:,:,3) = L;
end
return;
function Image = lab(Image,SrcSpace)
% Convert to CIE L*a*b* (CIELAB)
WhitePoint = [0.950456,1,1.088754];
switch SrcSpace
case 'lab'
   return;
case 'lch'
   % Convert CIE L*CH to CIE L*ab
   C = Image(:,:,2);
   Image(:,:,2) = cos(Image(:,:,3)*pi/180).*C;  % a*
   Image(:,:,3) = sin(Image(:,:,3)*pi/180).*C;  % b*
otherwise
   Image = xyz(Image,SrcSpace);  % Convert to XYZ
   % Convert XYZ to CIE L*a*b*
   
X = Image(:,:,1)/WhitePoint(1);
   Y = Image(:,:,2)/WhitePoint(2);
   Z = Image(:,:,3)/WhitePoint(3);
   fX = f(X);
   fY = f(Y);
   fZ = f(Z);
   Image(:,:,1) = 116*fY - 16;
   Image(:,:,2) = 500*(fX - fY);  % a*
   Image(:,:,3) = 200*(fY - fZ);  % b*
end
return;
function Image = luv(Image,SrcSpace)
% Convert to CIE L*u*v* (CIELUV)
WhitePoint = [0.950456,1,1.088754];
WhitePointU = (4*WhitePoint(1))./(WhitePoint(1) +
15*WhitePoint(2) + 3*WhitePoint(3));
WhitePointV = (9*WhitePoint(2))./(WhitePoint(1) +
15*WhitePoint(2) + 3*WhitePoint(3));
Image = xyz(Image,SrcSpace); % Convert to XYZ
Denom = Image(:,:,1) + 15*Image(:,:,2) + 3*Image(:,:,3); U = (4*Image(:,:,1))./(Denom + (Denom == 0));
V = (9*Image(:,:,2))./(Denom + (Denom == 0));
Y = Image(:,:,2)/WhitePoint(2);
L = 116*f(Y) - 16;
Image(:,:,1) = L; % L* Image(:,:,2) = 13*L.*(U - WhitePointU); % u* Image(:,:,3) = 13*L.*(V - WhitePointV); % v*
return;
function Image = lch(Image,SrcSpace)
% Convert to CIE L*ch
Image = lab(Image,SrcSpace); % Convert to CIE L*ab
H = atan2(Image(:,:,3),Image(:,:,2));
H = H*180/pi + 360*(H < 0);
Image(:,:,2) = sqrt(Image(:,:,2).^2 + Image(:,:,3).^2); %C
Image(:,:,3) = H;
%H
return;
