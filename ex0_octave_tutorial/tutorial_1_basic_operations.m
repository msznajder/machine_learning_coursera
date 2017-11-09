%% 1. BASIC OPERATIONS

5 + 6 % 11

3 - 2 % 1

5 * 8 % 40

1 / 2 % 0.5

2^6 % 64

1 == 2 % 0

1 ~= 2 % 1

1 && 0 % 0

1 || 0 % 1

xor(1, 0) % 1

% Comma chain of commands.
a = 1, b = 2, c = 3

%% 2. VARIABLES

a = 3

a = 3; % assignment with output supressed

s = "hi"

c = (3 >= 1) % 1

% Basic display.
a = pi
a % 3.1416

% More complex display.
disp(a) % 3.1416

% Display strings.
disp(sprintf("2 decimals: %0.2f", a)) % 2 decimals: 3.14

disp(sprintf("6 decimals: %0.6f", a)) % 2 decimals: 6 decimals: 3.141593

% Display long decimals.
format long
a % 3.141592653589793

% Display short decimals.
format short
a % 3.1416

%% 3. WORKING WITH DATA

% Check working directory.
pwd 
% '/Users/michalsznajder/Dropbox/github/machine_learning_coursera/week_2_octave_tutorial'

% Load example data set.
load("example_data.txt")

% Show variables in current workspace.
who
% Your variables are:
% 
% A             ans           example_data  sz            w             
% a             c             s             v             

% Show variables in current workspace with details.
whos
%  Name               Size               Bytes  Class      Attributes
% 
%   A                  3x3                   72  double               
%   a                  1x1                    8  double               
%   ans                1x85                 170  char                 
%   c                  1x1                    1  logical              
%   example_data      97x2                 1552  double               
%   s                  1x1                  132  string               
%   sz                 1x2                   16  double               
%   v                  1x4                   32  double               
%   w                  1x10000            80000  double 

% Display data.
example_data
%     6.1101   17.5920
%     5.5277    9.1302
%     8.5186   13.6620
%     7.0032   11.8540
%     5.8598    6.8233
%       ...

% Get data size.
size(example_data)
%     97     1

% Get rid of a variable.
clear A

% Assign data set range to variable.
v = example_data(1:10)
%     6.1100
%     5.5270
%     8.5180
%     7.0030
%     5.8590
%     8.3820
%     7.4760
%     8.5780
%     6.4860
%     5.0540

% Write data to file in binary format.
save example_data_output.mat v;

% Write data to file in human readable format.
save example_data_output.txt v -ascii

% Clear all variables in workspace.
clear

%% 4. VECTORS AND MATRICES

% Create a matrix.
A = [1 2; 3 4; 5 6]

%      1     2
%      3     4
%      5     6

% Create a row vector - 1x3 matrix.
v = [1, 2, 3]
%   1     2     3

% Create a column vector - 3x1 matrix.
v = [1; 2; 3]
%      1
%      2
%      3

% Create a vector starting from 1, with 0.1 step to 2.
v = 1:0.1:2
%   Columns 1 through 7
% 
%     1.0000    1.1000    1.2000    1.3000    1.4000    1.5000    1.6000
% 
%   Columns 8 through 11
% 
%     1.7000    1.8000    1.9000    2.0000

% Create a vector with range 1 to 6.
v = 1:6
%     1     2     3     4     5     6

% Create matrix filled with ones.
ones(2, 3)
%      1     1     1
%      1     1     1

% Create any value matrix.
2 * ones(2, 3)
%      2     2     2
%      2     2     2

% Create matrix filled with zeros.
zeros(1, 3)
%      0     0     0

% Create matrix filled with uniform distribution (0-1) random numbers.
rand(3, 3)
%     0.9134    0.2785    0.9649
%     0.6324    0.5469    0.1576
%     0.0975    0.9575    0.9706

% Create matrix filled with normal distribution (mean=1, std=0) random numbers.
randn(3, 3)
%    -1.2075    0.4889   -0.3034
%     0.7172    1.0347    0.2939
%     1.6302    0.7269   -0.7873

% Random matrix calculation.
w = -6 + sqrt(10) * (randn(1, 10000));

% Plot matrix histogram.
% hist(w, 50) % mean because we subtracted 5 from mean 0

% Create identity matrix.
eye(4)
%      1     0     0     0
%      0     1     0     0
%      0     0     1     0
%      0     0     0     1

% Get documentation content for a function.
help eye

% Size of a matrix.
sz = size(A)
%      3     2

% Size of matrix one dimention.
A = [1 2 3; 4 5 6; 7 8 9]
size(A, 1) % 3

% Length of a matrix.
length(A) % 3

% Length of a vector.
v = [1, 2, 3, 4]
length(v) % 4

% Get matrix row column element.
A = [1 2; 3 4; 5 6]
%      1     2
%      3     4
%      5     6
A(3, 2) % 6

% Get matrix row.
A(2, :) % rows, cols
%     3     4

% Get matrix column.
A(:, 2) % rows, cols
%      2
%      4
%      6

% Get first and third row all columns elements.
A([1 3], :) % rows, cols
%      1     2
%      5     6

% Assign matrix column to a COLUMN vector.
A(:, 2) = [10; 11; 12]
%      1    10
%      3    11
%      5    12

% Append column vector to a matrix.
A = [A, [100; 101; 102]]
%      1    10   100
%      3    11   101
%      5    12   102

% Flatten a matrix into a column vector.
A(:)
%      1
%      3
%      5
%     10
%     11
%     12
%    100
%    101
%    102

% Concatenate two matrices next to each other.
A = [1, 2; 3, 4; 5, 6]
%      1     2
%      3     4
%      5     6
B = [11, 12; 13, 14; 15, 16]
%     11    12
%     13    14
%     15    16
C = [A, B]
%      1     2    11    12
%      3     4    13    14
%      5     6    15    16

% Concatenate two matrices on top of each other.
D = [A; B]
%      1     2
%      3     4
%      5     6
%     11    12
%     13    14
%     15    16

%% 5. LINEAR ALGEBRA

% Create a matrix.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]
%      1     2     3
%      4     5     6
%      7     8     9
%     10    11    12

% Initilize a vactor.
v = [1; 2; 3]
%      1
%      2
%      3

% Get the dimensions of the matrix - m = rows and n = columns.
[m, n] = size(A)
% m = 4
% n = 3

% Get the dimensions of the matrix - tuple like solution.
dim_A = size(A)
%      4     3
     
% Get the dimensions of the vector.
dim_v = size(v)
%      3     1

% Get element of a matrix by specifying row and column.
A_23 = A(2, 3)
%      6

% Matrix element-wise addition.
A = [1, 2, 4; 5, 3, 2]
%      1     2     4
%      5     3     2
     
B = [1, 3, 4; 1, 1, 1]
%      1     3     4
%      1     1     1
     
add_AB = A + B
%      2     5     8
%      6     4     3

% Matrix element-wise subtraction.
sub_AB = A - B
%      0    -1     0
%      4     2     1

% Scalar matrix multiplication.
s = 2
%      2
mult_As = A * s
%      2     4     8
%     10     6     4

% Scalar matrix division.
div_As = A / s
%     0.5000    1.0000    2.0000
%     2.5000    1.5000    1.0000

% Scalar matrix addition.
add_As = A + s
%      3     4     6
%      7     5     4

% Matrix vector multiplication.
A = [1, 2, 3; 4, 5, 6;, 7, 8, 9]
%      1     2     3
%      4     5     6
%      7     8     9

v = [1; 1; 1]
%      1
%      1
%      1

Av = A * v
%     6
%     15
%     24

% Multiply matrices.
A = [1, 2; 3, 4; 5, 6]
%      1     2
%      3     4
%      5     6
     
B = [1; 2]
%      1
%      2
     
mult_AB = A * B % We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
%      5
%     11
%     17

% Multiply by identity matrix.
A = [1, 2; 4, 5]
%      1     2
%      4     5

I = eye(2)
%      1     0
%      0     1

IA = I * A
%      1     2
%      4     5
     
AI = A * I
%      1     2
%      4     5

% Matrix multiplication is not commutative.
B = [1, 2;, 0, 2]
%      1     2
%      0     2
     
AB = A * B
%      1     6
%      4    18
     
BA = B * A % AB != BA
%      9    12
%      8    10

% Transpose matrix.
A_trans = A'
%      1     4
%      2     5

% Inverse matrix.
A_inv = inv(A)
%    -1.6667    0.6667
%     1.3333   -0.3333
  
% A^(-1)*A?
A_invA = inv(A) * A
%      1     0
%      0     1

%% 6. COMPUTING ON DATA

A = [1, 2; 3, 4; 5, 6]
%      1     2
%      3     4
%      5     6
B = [11, 12; 13, 14; 15, 16]
%     11    12
%     13    14
%     15    16
C = [1, 1; 2, 2]
%      1     1
%      2     2
v = [1; 2; 3]
%      1
%      2
%      3

% Algebraic matrix multiplication.
A * C
%      5     5
%     11    11
%     17    17

% Elementwise matrix multiplication.
A .* B % In general . marks elementwise operations - like default in NumPy.
%     11    24
%     39    56
%     75    96

% Elementwise squaring.
A .^ 2
%      1     4
%      9    16
%     25    36

% Elementwise reciprocal 
1 ./ v
%     1.0000
%     0.5000
%     0.3333

1 ./ A
%     1.0000    0.5000
%     0.3333    0.2500
%     0.2000    0.1667

% Elementwise logarithm.
log(v)
%          0
%     0.6931
%     1.0986

% Elementwise exponentiation.
exp(v)
%     2.7183
%     7.3891
%    20.0855

% Elementwise absolute value.
abs(v)
%      1
%      2
%      3

% Negative values.
-v % -1 * v
%     -1
%     -2
%     -3

% Increment by 1 elementwise.
v + 1

% Transpose matrix.
A
%      1     2
%      3     4
%      5     6
A'
%      1     3     5
%      2     4     6
A''
%      1     2
%      3     4
%      5     6

% Get maximum value.
v
%      1
%      2
%      3
max(v) 
% 3

% Get maximum value and index.
[val, ind] = max(v)
% val =
%      3
% ind =
%      3

% Warning: max on matrix does column wise maximum.
A
%      1     2
%      3     4
%      5     6
max(A)
%      5     6

% Elementwise comparison.
v
%      1
%      2
%      3
v < 3
%    1
%    1
%    0

% Get vector indices for elements passing condition.
find(v < 3)
%      1
%      2

% Get magic square matrix - all rows, cols and diags sum to the same thing.
A = magic(3) % useful rather only to generate matrix
%      8     1     6
%      3     5     7
%      4     9     2

% Get matrix rows and columns of elements passing condition. 
[r, c] = find(A >= 7)
% r =
%      1
%      3
%      2
% c =
%      1
%      2
%      3

% Advice: use "help find" rather than memorize it all.

% Sum all elements up.
sum(v) 
% 6

% Multiply all elements up.
prod(v)
6

% Round down - floor.
floor(v)
%      1
%      2
%      3
     
% Round up - ceil.
ceil(v)
%      1
%      2
%      3

% Generate random 3x3 matrix.
rand(3)
%     0.7749    0.6375    0.3058
%     0.3371    0.4498    0.8428
%     0.2712    0.5063    0.0158
    
% Get elementwise maximum of two matrices.
max(rand(3), rand(3))
%     0.7172    0.5898    0.6525
%     0.8659    0.5473    0.5735
%     0.7188    0.8048    0.7822

% Get matrix column wise maximum.
A
%      8     1     6
%      3     5     7
%      4     9     2
max(A, [], 1) % 1 means to take max along first dimension of A
%      8     9     7

% or
max(A)
%   8     9     7

% Get matrix row wise maximum.
max(A, [], 2)
%      8
%      7
%      9

% Max in all matrix elements.
max(max(A))
% 9

% or
max(A(:))
% 9

% Sum matrix along columns.
sum(A, 1)
%     15    15    15

% Sum matrix along rows.
sum(A, 2)
%     15
%     15
%     15

% Zero non-diagonal matrix element.
A
%      8     1     6
%      3     5     7
%      4     9     2
A .* eye(3)
%      8     0     0
%      0     5     0
%      0     0     2

% Sum matrix diagonal elements.
sum(sum(A .* eye(3)))
% 15

% Invert matrix.
pinv(A)
%     0.1472   -0.1444    0.0639
%    -0.0611    0.0222    0.1056
%    -0.0194    0.1889   -0.1028

%% 7. CONTROL STATEMENTS

v = zeros(10, 1)
%      0
%      0
%      0
%      0
%      0
%      0
%      0
%      0
%      0
%      0

% for loop.
for i=1:10,
    v(i) = 2^i;
end;
%            2
%            4  
%            8
%           16
%           32
%           64
%          128
%          256
%          512
%         1024

% while loop.
i = 1
while i <= 5,
    v(i) = 100;
    i = i + 1;
end;
v
%          100
%          100
%          100
%          100
%          100
%           64
%          128
%          256
%          512
%         1024

% while with break.
i = 1
while true,
    v(i) = 999;
    i = i + 1;
    if i == 6,
        break
    end;
end;
v
%          999
%          999
%          999
%          999
%          999
%           64
%          128
%          256
%          512
%         1024

% if else structure
if v(1) == 1,
    disp("The value is one");
elseif v(1) == 2,
    disp("The value is two");
else
    disp("The value is not one or two.");
end;
% The value is not one or two.

%% 8. PLOTTING DATA

% Plotting gives great insight into data and what is going on.

% Plot one line.
t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
plot(t, y1)
y2 = cos(2*pi*4*t);

% Plot two lines on one plot.
plot(t, y1)
hold on;
plot(t, y2, "r")
xlabel("time")
ylabel("value")
legend("sin", "cos")
title("my plot")

% Save plot.
print -dpng myPlot.png

% plot help
help plot

% Close plot.
close

% Plot two figures separately.
figure(1); plot(t, y1);
figure(2); plot(t, y2); 
close 

% Subplots with grid - rows x cols.
subplot(1, 2, 1);
plot(t, y1);
subplot(1, 2, 2);
plot(t, y2);

% Set axis values.
axis([0.5 1 -1 1])

% Clear frame.
clf;

% Plot matrix of colors.
imagesc(A)
imagesc(A), colorbar, colormap gray;

%% 9. FUNCTIONS

% Define function.
% - each function is in separate file of the name same as the function name.
% - to use function from a file you have to move (cd) to place where function 
% script file is located.

% Defined function usage. 
squareThisNumber(2)
% 4

[a, b] = squareAndCubeThisNumber(2)
a % 4
b % 8

% Cost function usage.
% Say we have data set with three points: (1, 1), (2, 2), (3, 3)
% We want to define a function to compute the cost function J(theta)
% for different values of theta.
X = [1, 1; 1, 2; 1, 3] % design matrix x with coordinates of three training examples
%      1     1
%      1     2
%      1     3
y = [1; 2; 3] % y axis values
%      1
%      2
%      3
theta = [0;1] 
%      0
%      1

j = costFunctionJ(X, y, theta)
% 0 % theta = [0; 1] gives exactly 45 degree line so perfect fit and no cost

theta = [0;0] 
%      0
%      0
j = costFunctionJ(X, y, theta) 
%     2.3333 % now we are predicting 0 for everything (1^2 + 2^2 + 3^2) / (2*3

% Example function definition.
function y = squareThisNumber(x) % return value, function name
y = x^2;
end % function in a separate file does not have this end

% Function with returning two values.
function [y1, y2] = squareAndCubeThisNumber(x)
y1 = x^2;
y2 = x^3;
end

% In a script like this all functions definitions must be located at the
% end of the file.

% Example cost function.
function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing our training examples.
% y is the class labels

m = size(X, 1);                     % number of training examples
predictions = X * theta;            % predictions of hypothesis on all m examples
sqrErrors = (predictions - y).^2;   % squared errors

J = 1 / (2 * m) * sum(sqrErrors);
end
     