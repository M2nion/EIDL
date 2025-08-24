clear;
clc;
close all;

% I assume the incidence wave is always came from vacuum(air).
% So, n1 = 1.0;

R_TFSF = load('Reflection_my_code.txt');
T_TFSF = load('Transmission_my_code.txt');

n1 = 1.0;
n2 = sqrt(2.0); % This value can be changed if you use an other material.

R_analytic = zeros(length(T_TFSF),1);
T_analytic = zeros(length(T_TFSF),1);

for i = 1 : length(T_TFSF)
    R_analytic(i) = (abs((n2-n1)/(n2+n1)))^2;
    T_analytic(i) = 1 - R_analytic(i);
end

wvl = linspace(0.4, 0.7, length(T_TFSF));
wvl = wvl.';

figure(1)
plot(wvl*1000, T_analytic,'k','LineWidth', 2);
hold on
plot(wvl*1000, flipud(T_TFSF), 'bo', 'LineWidth',1);
title('eps = 2')
xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
ylabel('Transmittance','fontname','arial','fontsize',16);
axis([400 700 0 1.0])
legend('Analytic', 'Meep(TFSF)','location','southeast');
set(gca,'fontname','arial','fontsize',14)
set(gca,'linewidth',1);
legend boxoff

figure(2) 
plot(wvl*1000, R_analytic,'k','LineWidth', 2);
hold on
plot(wvl*1000, flipud(R_TFSF), 'bo', 'LineWidth',1);
title('eps = 2')
xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
ylabel('Reflectance','fontname','arial','fontsize',16);
axis([400 700 0 1.0])
legend('Analytic', 'Meep(TFSF)','location','northeast');
set(gca,'fontname','arial','fontsize',14)
set(gca,'linewidth',1);
legend boxoff