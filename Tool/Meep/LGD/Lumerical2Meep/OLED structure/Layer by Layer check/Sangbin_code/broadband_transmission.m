clear all;
clc;
close all;

T_1_R_ = zeros(200,1);
T_1_G_ = zeros(200,1);
T_1_B_ = zeros(200,1);
T_2_R_ = zeros(200,1);
T_2_G_ = zeros(200,1);
T_2_B_ = zeros(200,1);

% T_1_R = load("Transmission_1_R.txt");
% T_1_G = load("Transmission_1_G.txt");
% T_1_B = load("Transmission_1_B.txt");
% T_2_R = load("Transmission_2_R.txt");
% T_2_G = load("Transmission_2_G.txt");
% T_2_B = load("Transmission_2_B.txt");
% % T_1_R = load("Transmission_R.txt");
% % T_1_G = load("Transmission_G1.txt");
% % T_1_B = load("Transmission_B.txt");
% % T_2_R = load("Transmission_R_30.txt");
% % T_2_G = load("Transmission_G1_30.txt");
% % T_2_B = load("Transmission_B_30.txt");
% 
% for i = 1:200
%     T_1_R_(i,1) = T_1_R(201-i,1);
%     T_1_G_(i,1) = T_1_G(201-i,1);
%     T_1_B_(i,1) = T_1_B(201-i,1);
%     T_2_R_(i,1) = T_2_R(201-i,1);
%     T_2_G_(i,1) = T_2_G(201-i,1);
%     T_2_B_(i,1) = T_2_B(201-i,1);
% end

T_1_R = load("Transmission_R.txt");
T_1_G_1 = load("Transmission_G1.txt");
T_1_G_2 = load("Transmission_G2.txt");
T_1_B = load("Transmission_B.txt");
T_2_R = load("Transmission_R_30.txt");
T_2_G_1 = load("Transmission_G1_30.txt");
T_2_G_2 = load("Transmission_G2_30.txt");
T_2_B = load("Transmission_B_30.txt");

for i = 1:200
    T_1_R_(i,1) = T_1_R(201-i,1);
    T_1_G_(i,1) = T_1_G_1(201-i,1) + T_1_G_2(201-i,1);
    T_1_B_(i,1) = T_1_B(201-i,1);
    T_2_R_(i,1) = T_2_R(201-i,1);
    T_2_G_(i,1) = T_2_G_1(201-i,1) + T_2_G_2(201-i,1);
    T_2_B_(i,1) = T_2_B(201-i,1);
end

figure
x = linspace(1/380, 1/780, 200);
wvl = 1.0./x;
y1 = T_1_R_;
plot(wvl/1000,y1, "red")
title('Transmission_1')
xlabel('wavelength')
ylabel('optical efficiency')
hold on

y2 = T_1_G_;
plot(wvl/1000,y2, "green")

y3 = T_1_B_;
plot(wvl/1000,y3, "blue")

hold off

figure
x = linspace(1/380, 1/780, 200);
wvl=1.0./x;
y1 = T_2_R_;
plot(wvl/1000,y1, "red")
title('Transmission_3')
xlabel('wavelength')
ylabel('optical efficiency')
hold on

y2 = T_2_G_;
plot(wvl/1000,y2, "green")

y3 = T_2_B_;
plot(wvl/1000,y3, "blue")

hold off

% q = linspace(0, 0.5, 100);
% q2 = q.^2;
% E_tar = exp(-25*q2);
% 
% figure
% plot(q, E_tar)
