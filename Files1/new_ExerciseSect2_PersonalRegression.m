%My student number: r0867950定义一个数据集
d1=9;
d2=8;
d3=7;
d4=6;
d5=5;

%Part 1
load("Data_Problem1_regression.mat");
TNew = (d1*T1 + d2*T3 + d3*T3 + d4*T4 + d5*T5)/(d1+d2+d3+d4+d5);
%Training set训练一个模型
temp = datasample([X1 X2 TNew],1000,1);%每个样本一千个点
trainingX = temp(:,1:2).';%训练集
trainingY = temp(:,3).';
trainingP = con2seq(trainingX);
trainingT = con2seq(trainingY);
%Validation set 验证
temp = datasample([X1 X2 TNew],1000,1);
validationX = temp(:,1:2).';
validationY = temp(:,3).';
validationP = con2seq(validationX);
validationT = con2seq(validationY);
%Test set 测试
temp = datasample([X1 X2 TNew],1000,1);
testX = temp(:,1:2).';
testY = temp(:,3).';
testP = con2seq(testX);
testT = con2seq(testY);


%Plot training set surface 绘制表面
x = trainingX(1,:).';
y = trainingX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,trainingY.');
Z = f(X,Y);
figure
mesh(X,Y,Z) %interpolated
axis tight; hold on
plot3(X,Y,Z,'.','MarkerSize',10) %nonuniform
%%
%compare
netNum=6;
nets{1}=feedforwardnet(10,'trainlm');
nets{2}=feedforwardnet(15,'trainlm');
nets{3}=feedforwardnet(20,'trainlm');
nets{4}=feedforwardnet([5 5],'trainlm');
nets{5}=feedforwardnet([6 6],'trainlm');
nets{6}=feedforwardnet([7 7 7],'trainlm');


% Train the neural networks
for i = 1:6
    nets{i}.trainParam.epochs=1000;
    [nets{i}] = train(nets{i},trainingP,trainingT);
    simulationTraining{i}=sim(nets{i},trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    simulationValidation{i}=sim(nets{i},validationP);
    mseValidation{i} = mean((validationY-cell2mat(simulationValidation{i})).^2);
end

%draw a picture
% Plot the MSE values for each neural network
figure;
plot(1:6, [mseTraining{:}], '-o', 'LineWidth', 2);
hold on;
plot(1:6, [mseValidation{:}], '-o', 'LineWidth', 2);
title('Comparison of MSE for Different Neural Networks');
xlabel('Neural Network');
ylabel('MSE');
legend('Training Data', 'Validation Data', 'Location', 'Best');
%%

%Test set error
mseTest = mean((testY-cell2mat(sim(nets{6},testP))).^2); %modify

%Plot test set surface
x = testX(1,:).';
y = testX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,testY.');
Z = f(X,Y);
figure
mesh(X,Y,Z) %interpolated 
% 画图:3D Plot the surface of the test set
axis tight; hold on
%plot3(x,y,z,'.','MarkerSize',10) %nonuniform

%Plot NN surface :approximation given by the network.
% nets{6} 6最优
x=testX(1,:).';
y=testX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
outRows = size(X, 1);
outCols = size(Y, 2);
Z = zeros(outRows, outCols);
for row = 1:outRows
    for col = 1:outCols
        input = [X(row, col); Y(row, col)];
        simulated = sim(nets{6},input);%modify
        Z(row, col) = simulated;
    end
end
figure
mesh(X,Y,Z,'FaceAlpha',0.7);
axis tight; hold on
%plot3(x,y,trainingY.','.','MarkerSize',10,'DisplayName','Training') %Training set
%plot3(validationX(1,:).',validationX(2,:).',validationY.','.','MarkerSize',10,'DisplayName','Validation') %Validation set

%plot3(X1.',X2.',TNew.','.','MarkerSize',15,'DisplayName','All'); %All points


%Plot error surface
x = testX(1,:).';
y = testX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,(testY - cell2mat(sim(nets{6},testP))).^2.');%modify
Z = f(X,Y);
figure;
mesh(X,Y,Z); %interpolated
axis tight; hold on;
contour3(X,Y,Z,100); %Error curves
plot3(trainingX(1,:).',trainingX(2,:).',trainingY.','.','MarkerSize',10); %nonuniform
hold off;