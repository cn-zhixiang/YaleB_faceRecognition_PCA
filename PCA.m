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
## @deftypefn {Function File} {@var{retval} =} PCA (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 王志翔 <wangzhixiang@wangzhixiangdeMacBook-Air.local>
## Created: 2016-12-18

function [COEFF, SCORE, latent] = PCA (X)

[m n] = size(X);
mu = mean(X, 1);
X_zero_mu = double(X - repmat(mu, m, 1));
sigma = 1/m* X_zero_mu' * X_zero_mu;

[U S V] = svd(sigma);
COEFF = U;
SCORE = X_zero_mu*COEFF;
latent = diag(S);

endfunction
