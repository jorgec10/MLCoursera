1 == 1
1~=2
1 &&0  % And
1 || 0 % or
xor(1,0)
PS1('>> ') % change prompt
a = 3; % ; supress output
disp(a); % show var
disp(sprintf('2 decimals: %0.2f', a))
format long % change default to show all decimals
format short % change default to show 4 decimals
A = [1 2; 3 4; 5 6]
A = [1 2;
3 4;
5 6]
v = [1 2 3]
v = 1:6
v = 1:0.1:2
ones(2,3)
C = 2*ones(2,3)
w = ones(1,3)
w = zeros(1,3)
w = rand(1,3)
rand(3,3)
randn(1,3) % gaussian
%w = -6 + sqrt(10)*(randn(1,10000))
%hist(w)
%hist(w,50)
eye(4) % identity matrix
eye(3)
% help eye
% help rand
% help help

size(A)
sz = size(A)
size(sz)
size(A,1) % rows
size(A,2) % columns
v = [1 2 3 4]
length(v)
length(A) % longest dimension

%% Moving data

pwd % current dir
load featuresX.dat
load priceY.dat
load('featuresX.dat')

who % variables in the current scope
featuresX
size(featuresX)
size(priceY)
whos %detail view of variables

v = priceY(1:10) % first 10 elements of priceY
save hello.mat v; % saves v in hello.mat (binary)
clear % delete variables
load hello.mat % load again v
save hello.txt -ascii v; % save as text (ASCII)

A(3,2) % element in row 3, column 2
A(2,:) % all elements in row 2
A([1 3],:) % all elements of A whose first index is 1 or 3
A(:,2) = [10;11;12] % substitute column
A = [A, [100;101;102]] % append a new column
A(:) % put all elements of A into a single column vector

C = [A B] % concatenate two matrices, A left, B right
C = [A;B] % concatenate, A top, B bottom


%% Computing on data
A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [1 1; 2 2]

A * C
A .* C % elementwise multiplications A11*C11, etc
A .^ C % elementwise pow
v = [1; 2; 3]
1 ./ v
1 ./ A % elementwise inverse
log(v) % elementwise log
exp(v) % exponentiation
abs(v) % absolute value
-v 
v + ones(length(v),1) % increment values from v by 1
length(v)
ones(3,1)
v + ones(3,1)
v + 1 % the same

A' % transpose
(A')'
val = max(a)
[val, ind] = max(a) % max and its position
max(A) % returns column with max

a < 3 % compares element by element
find(a < 3) % elements that are less than 3

help magic
A = magic(3) % generate 3x3 magic matrix
[r,c] = find(A >= 7) % finds elements of A 
help find

sum(a) % adds all elements
prod(a) % prod all elements
floor(a) % round down
ceil(a) % round up
max(rand(3), rand(3))
max(A,[],1) % columnwise maximum
max(A,[],2) % rowwise maximum
max(max(A)) % max of matrix
max(A(:)) % max of matrix, convert to vector and max
sum(A,1) % per column sum
sum(A,2) % per row sum
eye(9)
A .* eye % elementwise product
sum(sum(A .* eye(9))) % sum of the diagonal
flipupd(eye(9)) % the other diagonal
A = magic(3)
pinv(A) % inverse of A
pinv(A) * A % identity


%% Ploting data


t = [0:0.01:0.98];
t
y1 = sin(2*pi*4*t)
plot(t,y1); % x t, y y1
y2 = cos(2*pi*4*t);
plot(t,y2);
hold on; % keep the last plot
plot(t,y2,'r') % plot above in red
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('my plot')
print -dpng 'myPlot.png'
close % delete figure
figure(1); plot(t,y1);
figure(2); plot(t,y2); % open 2 windows
subplot(1,2,1) % subdivides plot in a 1x2 frid, access first element
plot(t,y1)
subplot(1,2,2)
plot(t,y2)
axis([0.5 1 -1 1]) % x 0.5-1 y -1 1
clf % clear figure
A = magic(5)
imagesc(A)
imagesc(A), colorbar, colormap gray;


%% Control statements
for i=1:10
	v(i) = 2^i;
end;

indices=1:10
for i=indices
	v(i) = 2^i;
end;

i=1;
while i <= 5,
	v(i) = 500;
	i = i+1;
end;

while true,
	v(i) = 999;
	i = i+1;
	if i == 6,
		break;
	elseif v(1) == 2,
		disp('2');
	else
		disp('other')
	end;
end;

%% inside a separate file
function output = myFun(input)
%myFun - Description
%
% Syntax: output = myFun(input)
%
% Long description
	output = input^2
	
end

%% call: myFun(5)
addpath('/home/jorge')

% return two values
function [y1,y2] = fun2(x)
end

[a,b] = fun2(5)

X = [1 1; 1 2; 1 3]
y = [1; 2; 3]
theta = [0;1];
j = costFunctionJ(X, y, theta)

function J = costFunctionJ(X, y, theta)
	% X is the desing matrix containing our training examples
	% y us the class labels

	% number of training examples
	m = size(X,1);
	% predictions of hypothesis on all m examples
	predictions = X*theta;
	% squared errors
	sqrErrors = (predictions-y).^2;

	J = 1/(2*m) * sum(sqrErrors);
end



%% Vectorization

% Unvectorized implementation
prediction = 0.0;
for j = 1:n+1,
	prediction = prediction + theta(j) * x(j)
end;

% Vectorized implementation
prediction = theta' * x

% Gradient descent
% Vectorized implementation


