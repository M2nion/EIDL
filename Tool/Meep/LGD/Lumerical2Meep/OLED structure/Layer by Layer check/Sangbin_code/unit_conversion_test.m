clc;
close all;
clear;
c = 2.99792e8;	        % speed of light [m/s]

option = 0;

eps_bkgd = 2+0.001i;
eps_lorentz = 1e-02;

data_eps_re=load('Si_eps_re.txt');
data_eps_im=load('Si_eps_im.txt');

%% Red
delta_o_red = 2e14;
omega_o_red = 2*pi*c/0.4e-6;
omega_o1_red = 2*pi*c/0.425e-6;
omega_o2_red = 2*pi*c/0.45e-6;
omega_o3_red = 2*pi*c/0.475e-6;
omega_o4_red = 2*pi*c/0.5e-6;
omega_o5_red = 2*pi*c/0.525e-6;
omega_o6_red = 2*pi*c/0.55e-6;


a = pi*2*2.99792e14; % Meep 1um -> omega % make sure "divided by 2pi"
meep_delta_o_red = (2*delta_o_red/a);
meep_freq_o_red = omega_o_red/a;
meep_freq_o1_red = omega_o1_red/a;
meep_freq_o2_red = omega_o2_red/a;
meep_freq_o3_red = omega_o3_red/a;
meep_freq_o4_red = omega_o4_red/a;
meep_freq_o5_red = omega_o5_red/a;
meep_freq_o6_red = omega_o6_red/a;

%% Green

delta_o_green = 2e14;
omega_o_green = 2*pi*c/0.4e-6;
omega_o1_green = 2*pi*c/0.425e-6;
omega_o2_green = 2*pi*c/0.45e-6; 
omega_o3_green = 2*pi*c/0.475e-6; 
omega_o4_green = 2*pi*c/0.65e-6;
omega_o5_green = 2*pi*c/0.675e-6;
omega_o6_green = 2*pi*c/0.7e-6;

meep_delta_o_green = (2*delta_o_green/a);
meep_freq_o_green = omega_o_green/a;
meep_freq_o1_green = omega_o1_green/a;
meep_freq_o2_green = omega_o2_green/a;
meep_freq_o3_green = omega_o3_green/a;
meep_freq_o4_green = omega_o4_green/a;
meep_freq_o5_green = omega_o5_green/a;
meep_freq_o6_green = omega_o6_green/a;

%% Blue

delta_o_blue = 2e14;
omega_o_blue = 2*pi*c/0.55e-6;
omega_o1_blue = 2*pi*c/0.575e-6;
omega_o2_blue = 2*pi*c/0.6e-6;
omega_o3_blue = 2*pi*c/0.625e-6;
omega_o4_blue = 2*pi*c/0.65e-6;
omega_o5_blue = 2*pi*c/0.675e-6;
omega_o6_blue = 2*pi*c/0.7e-6;

meep_delta_o_blue = (2*delta_o_blue/a);
meep_freq_o_blue = omega_o_blue/a;
meep_freq_o1_blue = omega_o1_blue/a;
meep_freq_o2_blue = omega_o2_blue/a;
meep_freq_o3_blue = omega_o3_blue/a;
meep_freq_o4_blue = omega_o4_blue/a;
meep_freq_o5_blue = omega_o5_blue/a;
meep_freq_o6_blue = omega_o6_blue/a;


% f = linspace(1.0/0.4, 1.0/0.7, 1000);
% lambda = 1./f;

lambda = linspace(0.4, 0.7, 1000);
f = 1./lambda;

eps_red = eps_bkgd + (5*eps_lorentz*meep_freq_o_red.^2)./(meep_freq_o_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (4*eps_lorentz*meep_freq_o1_red.^2)./(meep_freq_o1_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o2_red.^2)./(meep_freq_o2_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o3_red.^2)./(meep_freq_o3_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o4_red.^2)./(meep_freq_o4_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o5_red.^2)./(meep_freq_o5_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o6_red.^2)./(meep_freq_o6_red.^2-1i*meep_delta_o_red.*f-(f).^2);
eps_green = eps_bkgd + (5*eps_lorentz*meep_freq_o_green.^2)./(meep_freq_o_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (4*eps_lorentz*meep_freq_o1_green.^2)./(meep_freq_o1_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (3*eps_lorentz*meep_freq_o2_green.^2)./(meep_freq_o2_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (3*eps_lorentz*meep_freq_o3_green.^2)./(meep_freq_o3_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (3*eps_lorentz*meep_freq_o4_green.^2)./(meep_freq_o4_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (4*eps_lorentz*meep_freq_o5_green.^2)./(meep_freq_o5_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (5*eps_lorentz*meep_freq_o6_green.^2)./(meep_freq_o6_green.^2-1i*meep_delta_o_green.*f-(f).^2);
eps_blue = eps_bkgd + (3*eps_lorentz*meep_freq_o_blue.^2)./(meep_freq_o_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o1_blue.^2)./(meep_freq_o1_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o2_blue.^2)./(meep_freq_o2_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (4*eps_lorentz*meep_freq_o3_blue.^2)./(meep_freq_o3_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o4_blue.^2)./(meep_freq_o4_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o5_blue.^2)./(meep_freq_o5_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (4*eps_lorentz*meep_freq_o6_blue.^2)./(meep_freq_o6_blue.^2-1i*meep_delta_o_blue.*f-(f).^2);

%% Lumerical

wvl = linspace(0.4e-6, 0.7e-6, 1066);
Lumerical_f = c./wvl;

% Lumerical_f = linspace(c./0.7e-6, c./0.4e-6, 1066);
% wvl = c./Lumerical_f;

Lumerical_eps_red = eps_bkgd + (5*eps_lorentz*omega_o_red.^2)./(omega_o_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o1_red.^2)./(omega_o1_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o2_red.^2)./(omega_o2_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o3_red.^2)./(omega_o3_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o4_red.^2)./(omega_o4_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o5_red.^2)./(omega_o5_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o6_red.^2)./(omega_o6_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2);
Lumerical_eps_green = eps_bkgd + (5*eps_lorentz*omega_o_green.^2)./(omega_o_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o1_green.^2)./(omega_o1_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o2_green.^2)./(omega_o2_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o3_green.^2)./(omega_o3_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o4_green.^2)./(omega_o4_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o5_green.^2)./(omega_o5_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (5*eps_lorentz*omega_o6_green.^2)./(omega_o6_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2);
Lumerical_eps_blue = eps_bkgd + (3*eps_lorentz*omega_o_blue.^2)./(omega_o_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o1_blue.^2)./(omega_o1_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o2_blue.^2)./(omega_o2_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o3_blue.^2)./(omega_o3_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o4_blue.^2)./(omega_o4_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o5_blue.^2)./(omega_o5_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o6_blue.^2)./(omega_o6_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2);


% figure(1)
% if option == 1
% 
%     plot(wvl.*10^9, real(Lumerical_eps_red),'ko','LineWidth', 1); % c./f.*10^(9)
%     hold on;
%     plot(lambda*1000,real(eps_red),'r-','LineWidth', 2);
%     title('Red color-filter')
%     hold on;
%     xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
%     ylabel('Re(eps)','fontname','arial','fontsize',16);
%     legend('Lumerical unit','Meep unit','location','southeast');
%     set(gca,'fontname','arial','fontsize',14)
%     set(gca,'linewidth',1);
%     legend boxoff
% 
% else
% 
%     plot(wvl.*10^9, imag(Lumerical_eps_red),'ko','LineWidth', 1); % c./f.*10^(9)
%     hold on;
%     plot(lambda*1000, imag(eps_red),'r-','LineWidth', 2);
%     title('Red color-filter')
%     hold on;
%     xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
%     ylabel('Im(eps)','fontname','arial','fontsize',16);
%     legend('Lumerical unit','Meep unit','location','northeast');
%     set(gca,'fontname','arial','fontsize',14)
%     set(gca,'linewidth',1);
%     legend boxoff
% 
% end
% 
% figure(2)
% if option == 1
% 
%     plot(wvl.*10^9, real(Lumerical_eps_green),'ko','LineWidth', 1); % c./f.*10^(9)
%     hold on;
%     plot(lambda*1000,real(eps_green),'g-','LineWidth', 2);
%     title('Green color-filter')
%     hold on;
%     xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
%     ylabel('Re(eps)','fontname','arial','fontsize',16);
%     legend('Lumerical unit','Meep unit','location','southeast');
%     set(gca,'fontname','arial','fontsize',14)
%     set(gca,'linewidth',1);
%     legend boxoff
% 
% else
% 
%     plot(wvl.*10^9, imag(Lumerical_eps_green),'ko','LineWidth', 1); % c./f.*10^(9)
%     hold on;
%     plot(lambda*1000, imag(eps_green),'g-','LineWidth', 2);
%     title('Green color-filter')
%     hold on;
%     xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
%     ylabel('Im(eps)','fontname','arial','fontsize',16);
%     legend('Lumerical unit','Meep unit','location','northeast');
%     set(gca,'fontname','arial','fontsize',14)
%     set(gca,'linewidth',1);
%     legend boxoff
% 
% end
% 
% figure(3)
% if option == 1
% 
%     plot(wvl.*10^9, real(Lumerical_eps_blue),'ko','LineWidth', 1); % c./f.*10^(9)
%     hold on;
%     plot(lambda*1000,real(eps_blue),'b-','LineWidth', 2);
%     title('Blue color-filter')
%     hold on;
%     xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
%     ylabel('Re(eps)','fontname','arial','fontsize',16);
%     legend('Lumerical unit','Meep unit','location','southeast');
%     set(gca,'fontname','arial','fontsize',14)
%     set(gca,'linewidth',1);
%     legend boxoff
% 
% else
% 
%     plot(wvl.*10^9, imag(Lumerical_eps_blue),'ko','LineWidth', 1); % c./f.*10^(9)
%     hold on;
%     plot(lambda*1000, imag(eps_blue),'b-','LineWidth', 2);
%     title('Blue color-filter')
%     hold on;
%     xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
%     ylabel('Im(eps)','fontname','arial','fontsize',16);
%     legend('Lumerical unit','Meep unit','location','southeast');
%     set(gca,'fontname','arial','fontsize',14)
%     set(gca,'linewidth',1);
%     legend boxoff
% 
% end

%% SiO2 

f0_sio2_meep = 1.0/0.103320160833333;
gamma_sio2_meep = 1.0/12.3984193000000;
sigma_sio2_meep = 1.12;

eps_SiO2 = 1.0 + (sigma_sio2_meep*f0_sio2_meep^2)/(f0_sio2_meep^2 - )

%% c-Si

eps_cSi = 1.0 + (8.0*3.64^2)./(3.64^2 - f.^2) + (2.85*2.76^2)./(2.76^2 - 1.0i*2*0.063.*f - f.^2) + (-0.107*1.73^2)./(1.73^2 - 1.0i*2*2.5.*f - f.^2);
Lumerical_eps_cSi = data_eps_re(:,2) + data_eps_im(:,2)*1.0i;
Lumerical_eps_cSi = Lumerical_eps_cSi.';

figure(4)
plot(linspace(700,400,1066), real(Lumerical_eps_cSi),'k-','LineWidth', 2);
hold on 
plot(lambda*1000, real(eps_cSi),'c-','LineWidth', 2);
title('Crystalline Si')
xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
ylabel('Re(eps)','fontname','arial','fontsize',16);
legend('Lumerical','Meep','location','northeast');
set(gca,'fontname','arial','fontsize',14)
set(gca,'linewidth',1);
legend boxoff

figure(5)
plot(linspace(700,400,1066), imag(Lumerical_eps_cSi),'k-','LineWidth', 2);
hold on 
plot(lambda*1000, imag(eps_cSi),'c-','LineWidth', 2);
title('Crystalline Si')
xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
ylabel('Im(eps)','fontname','arial','fontsize',16);
legend('Lumerical','Meep','location','northeast');
set(gca,'fontname','arial','fontsize',14)
set(gca,'linewidth',1);
legend boxoff

filename_meep_real_cSi = "Crystalline_Silicon_real_eps_meep_unit.txt";
resultm = [f;real(eps_cSi)];
writematrix(resultm.',filename_meep_real_cSi);

% filename_lumerical_real_cSi = "Crystalline_Silicon_real_eps_lumerical.txt";
% resultm = [Lumerical_f;real(Lumerical_eps_cSi)];
% writematrix(resultm.',filename_lumerical_real_cSi);

filename_meep_imag_cSi = "Crystalline_Silicon_imag_eps_meep_unit.txt";
resultm = [f;imag(eps_cSi)];
writematrix(resultm.',filename_meep_imag_cSi);

% filename_lumerical_imag_cSi = "Crystalline_Silicon_imag_eps_lumerical.txt";
% resultm = [Lumerical_f;imag(Lumerical_eps_cSi)];
% writematrix(resultm.',filename_lumerical_imag_cSi);
