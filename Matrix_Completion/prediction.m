load('/home/yyldzxx/coupon_user.csv')
nuser = max(mcdata(:, 1));
ncoupon = max(mcdata(:, 2));
w0 = rand(nuser, 5) .* 2 -1;
h0 = rand(5, ncoupon) .* 2 -1;
[w,h,recmat] = matrix_cmplt(w0, h0, mcdata, 2.5, 0.1);
save para.mat

