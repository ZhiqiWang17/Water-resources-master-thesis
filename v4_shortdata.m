%inport the data
x = input_short_calc;
y = target_short;
m = length (y);

%% visualization of the data
% chl_train = y(1,:);
% SPM_train = y(2,:);
% T_train = y(3,:);

histogram(y(1,:),10);
histogram(y(2,:),10);
histogram(y(3,:),10);

histogram(x(1,:),10);
histogram(x(2,:),10);
histogram(x(3,:),10);

%relations between input-output for each parameter
plot(x(1,:),y(1,:),'o');
title('Correlation of RS & in-situ chlorophylla data');
xlabel('RS chlorophyll a [ug/l]'); 
ylabel('In-situ chlorophyll a [ug/l]'); 

plot(x(2,:),y(2,:),'o');
title('Correlation of RS SPM & in-situ turbidity data');
xlabel('RS SPM [g/m3]'); 
ylabel('In-situ Turbidity [NTU]'); 

plot(x(3,:),y(3,:),'o');
title('Correlation of RS & in-situ temperature data');
xlabel('RS Temperature [℃]'); 
ylabel('In-situ Temperature [℃]');

%correlation for input-output
plot(x,y,'o');hold on;
plot(0:100,0:100); 
title('Correlation of RS & in-situ data');
xlabel('RS input data'); 
ylabel('In-situ target data');hold off; 

%correlation between input parameters
%chl-a with SPM/turbidity
plot(x(2,:),x(3,:),'o');
title('Correlation of input SPM and temperature data');
xlabel('RS SPM [g/m3]'); 
ylabel('RS Temperature [℃]'); 


%REGRESSION LINE & R^2
mdl = fitlm(x(1,:),y(1,:));
mdl.Rsquared.Ordinary


%% Normalize the feature and transform the output
y2 = log(1+y);

histogram(y2(1,:),10);
histogram(y2(2,:),10);
histogram(y2(3,:),10);
% use log transformation to initial target to make the data more uniform
% distributed, less centered around very low values 

for i = 1:3
    x2(i,:) = (x(i,:)-min(x(i,:)))/(max(x(i,:))-min(x(i,:)));
    %inout data to remove the min value and divide the range of data
end

histogram(x2(1,:),10);
histogram(x2(2,:),10);
histogram(x2(3,:),10);
%normalized input data is between 0-1, to make sure all the input have 
%same weight, and to accelate the training.

%relations between input-output after transformation
plot(x2(1,:),y2(1,:),'o');
title('Line Plot of Sine and Cosine Between -2\pi and 2\pi')

plot(x2(2,:),y2(2,:),'o');
plot(x2(3,:),y2(3,:),'o');

mdl = fitlm(x2(1,:),y2(1,:));
mdl = fitlm(x2(2,:),y2(2,:));
mdl = fitlm(x2(3,:),y2(3,:));
mdl.Rsquared.Ordinary

%% train an ANN
%create a shallow network whith one hidden layer, the neurons number is
%crutial
hiddenLayerSize = 7;
%trainFcn = 'trainlm';
net = fitnet(hiddenLayerSize);
%net.layers{1}.transferFcn = 'logsig';

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 30/100;
%[trainInd,valInd,testInd] = divideblock(203,0.5,0.2,0.3);

[net,tr] = train(net,x2,y2);

%% Performance of the ANN model
%compare the prediction of the model with the true output value for
%training and validation set

yTrain = exp(net(x2(:,tr.trainInd)))-1;% provide the value of the feature for the training set
yTrainTrue = exp(y2(:,tr.trainInd))-1;
RMSE_train = sqrt(mean((yTrain - yTrainTrue).^2));

yVal = exp(net(x2(:,tr.valInd)))-1;% provide the value of the feature for the validation set
yValTrue = exp(y2(:,tr.valInd))-1;
RMSE_val = sqrt(mean((yVal - yValTrue).^2));

%% visualize the prediction form the ANN model

plot(yTrain,yTrainTrue,'xk'); hold on;% x-axis is predicted chance of
%chl/turb/T by model，y-axis is true chance of chl/turb/T
plot(yVal,yValTrue,'ok'); hold on;
%these point tell how well the model is generalizing
plot(yTest,yTestTrue,'vr'); hold on;
%the far the point to the dianoge line, the less good model is
plot(0:100,0:100); hold off;

title('Correlation of simulated RS & in-situ data of model ANN2');
xlabel('Simulated RS input data'); 
ylabel('In-situ target data'); 
%% get the trained test data and compare with original

yTest = exp(net(x2(:,tr.testInd)))-1;%% get the trained test data and compare with original
yTestTrue = exp(y2(:,tr.testInd))-1;
RMSE_test = sqrt(mean((yTest - yTestTrue).^2));

plot(yTestTrue,yTest,'x'); hold on;
plot(0:100,0:100); hold off;


mdl = fitlm(yTest(1,:),yTestTrue(1,:));
mdl = fitlm(yTest(2,:),yTestTrue(2,:));
mdl = fitlm(yTest(3,:),yTestTrue(3,:));
mdl.Rsquared.Ordinary

%%
x = ANN1_simulated_RS;
y = ANN1_insitu;
%relations between input-output for each parameter
plot(x(1,:),y(1,:),'o');
title('Correlation of simulated & in-situ chlorophylla data');
xlabel('RS chlorophyll a [ug/l]'); 
ylabel('In-situ chlorophyll a [ug/l]'); 

plot(x(2,:),y(2,:),'o');
title('Correlation of RS SPM & in-situ turbidity data');
xlabel('RS SPM [g/m3]'); 
ylabel('In-situ Turbidity [NTU]'); 

plot(x(3,:),y(3,:),'o');
title('Correlation of RS & in-situ temperature data');
xlabel('RS Temperature [℃]'); 
ylabel('In-situ Temperature [℃]');

%REGRESSION LINE & R^2
mdl = fitlm(x(3,:),y(3,:));
mdl.Rsquared.Ordinary