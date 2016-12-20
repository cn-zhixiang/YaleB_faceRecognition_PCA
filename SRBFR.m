## Copyright (C) 2016 王志翔
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} SRBFR (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 王志翔 <wangzhixiang@wangzhixiangdeMacBook-Air.local>
## Created: 2016-12-18

function [ACCURACY] = SRBFR (numTrainSamples, filePath)

time_begin = time;

number_person = 39;
number_training_per_person = numTrainSamples;
number_test_per_person = 59-number_training_per_person;
number_images_per_person = number_training_per_person+number_test_per_person;

height = 192;
width = 168;
scale = 4;
faces_training = zeros(1,width/scale*height/scale);
faces_test = zeros(1,width/scale*height/scale);
labels_training = zeros(1,1);
labels_test = zeros(1,1);

for i = 1:number_person
  if i==14
    continue;
  end
  dir_persion_i = sprintf('%s/yaleB%02d/',filePath, i);
  filenames = dir(strcat(dir_persion_i,'*[05].pgm'));
  number_pgm = length(filenames);
  fprintf('number_pgm is %d\n', number_pgm);
  index = randperm(number_pgm);
  
  for j=1:number_images_per_person
    image = imread(strcat(dir_persion_i, filenames(index(j)).name))(1:scale:end, 1:scale:end);
    [rows cols] = size(image);
    %load training and test samples
    if j <= number_training_per_person
      faces_training = [faces_training; reshape(image, 1, rows*cols/1)];
      labels_training = [labels_training; i];
    else
      faces_test = [faces_test; reshape(image, 1, rows*cols/1)];
      labels_test = [labels_test; i];
    end
  end
end

faces_training = double(faces_training(2:end,:));
faces_test = double(faces_test(2:end,:));
labels_training = labels_training(2:end);
labels_test = labels_test(2:end);

[COEFF, SCORE, latent] = PCA(faces_training);

rate = 0.95;
sum_info = sum(latent);
sum_info_current = latent(1);
n = length(latent);
k = 1;
for i=2:n
  rate_1 = sum_info_current/sum_info;
  sum_info_current = sum_info_current + latent(i);
  rate_2 = sum_info_current/sum_info;
  if  rate_1 <= rate && rate_2 >= rate
    k = i;
    break;
  end
end
fprintf("k is %d\n", k);

A = COEFF(:, 1:k);
lambda = 0.00001;

numTrainSamples = size(faces_training, 1);
x_all_training = zeros(numTrainSamples, k);
for i=1:numTrainSamples
  y = faces_training(i,:)';
  x_init = A'*y;
  x_i = feature_sign(A, y, lambda, x_init);
  x_all_training(i,:) = x_i;
end

x_mean = mean(abs(x_all_training), 1);
x_norm = x_all_training./repmat(x_mean, numTrainSamples, 1);

numTestSamples = size(faces_test, 1);
labels_pred = zeros(numTestSamples, 1);
face_mean = mean(faces_training, 1);
for i=1:numTestSamples
  y = (faces_test(i,:)-face_mean)';
  x_init = A'*y;
  x_i = feature_sign(A, y, lambda, x_init);

  x_i_norm = x_i./x_mean';
  delta_x = x_norm - repmat(x_i_norm', numTrainSamples, 1);
  L2 = sum(delta_x .* delta_x, 2);
  [value idx] = min(L2);
  labels_pred(i) = labels_training(idx);
end

ACCURACY = sum(round(labels_pred)==round(labels_test))/numTestSamples;
printf('ACCURACY equals to %.4f\n', ACCURACY);
time_diff = time-time_begin;
printf('Program took %.2f seconds\n', time_diff);

endfunction
