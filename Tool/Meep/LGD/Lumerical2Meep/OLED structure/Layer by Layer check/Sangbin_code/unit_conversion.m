clc;
close all;
clear;
c = 2.99792e8;	        % speed of light [m/s]

option = 1;

eps_bkgd = 2+0.001i;
eps_lorentz = 1e-02;

%% Red
delta_o_red = 2.0e14;
omega_o_red = 2*pi*c/0.4e-6;
omega_o1_red = 2*pi*c/0.425e-6;
omega_o2_red = 2*pi*c/0.45e-6;
omega_o3_red = 2*pi*c/0.475e-6;
omega_o4_red = 2*pi*c/0.5e-6;
omega_o5_red = 2*pi*c/0.525e-6;
omega_o6_red = 2*pi*c/0.55e-6;


a = pi*2*2.99792e14; % Meep 1um c -> 2.99792e14 -> omega calculation, 2pi*~ -> frequency calculation
meep_delta_o_red = (2*delta_o_red/a);
meep_freq_o_red = omega_o_red/a;
meep_freq_o1_red = omega_o1_red/a;
meep_freq_o2_red = omega_o2_red/a;
meep_freq_o3_red = omega_o3_red/a;
meep_freq_o4_red = omega_o4_red/a;
meep_freq_o5_red = omega_o5_red/a;
meep_freq_o6_red = omega_o6_red/a;

%% Green

delta_o_green = 2.0e14;
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

delta_o_blue = 2.0e14;
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


lambda = linspace(0.4, 0.7, 50);
f = 1./lambda;

% omega calculation
% eps_red = eps_bkgd + (5*eps_lorentz*meep_freq_o_red.^2)./(meep_freq_o_red.^2-1i*meep_delta_o_red*2*pi*f-(2*pi*f).^2) + (4*eps_lorentz*meep_freq_o1_red.^2)./(meep_freq_o1_red.^2-1i*meep_delta_o_red*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o2_red.^2)./(meep_freq_o2_red.^2-1i*meep_delta_o_red*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o3_red.^2)./(meep_freq_o3_red.^2-1i*meep_delta_o_red*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o4_red.^2)./(meep_freq_o4_red.^2-1i*meep_delta_o_red*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o5_red.^2)./(meep_freq_o5_red.^2-1i*meep_delta_o_red*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o6_red.^2)./(meep_freq_o6_red.^2-1i*meep_delta_o_red*2*pi*f-(2*pi*f).^2);
% eps_green = eps_bkgd + (5*eps_lorentz*meep_freq_o_green.^2)./(meep_freq_o_green.^2-1i*meep_delta_o_green*2*pi*f-(2*pi*f).^2) + (4*eps_lorentz*meep_freq_o1_green.^2)./(meep_freq_o1_green.^2-1i*meep_delta_o_green*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o2_green.^2)./(meep_freq_o2_green.^2-1i*meep_delta_o_green*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o3_green.^2)./(meep_freq_o3_green.^2-1i*meep_delta_o_green*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o4_green.^2)./(meep_freq_o4_green.^2-1i*meep_delta_o_green*2*pi*f-(2*pi*f).^2) + (4*eps_lorentz*meep_freq_o5_green.^2)./(meep_freq_o5_green.^2-1i*meep_delta_o_green*2*pi*f-(2*pi*f).^2) + (5*eps_lorentz*meep_freq_o6_green.^2)./(meep_freq_o6_green.^2-1i*meep_delta_o_green*2*pi*f-(2*pi*f).^2);
% eps_blue = eps_bkgd + (3*eps_lorentz*meep_freq_o_blue.^2)./(meep_freq_o_blue.^2-1i*meep_delta_o_blue*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o1_blue.^2)./(meep_freq_o1_blue.^2-1i*meep_delta_o_blue*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o2_blue.^2)./(meep_freq_o2_blue.^2-1i*meep_delta_o_blue*2*pi*f-(2*pi*f).^2) + (4*eps_lorentz*meep_freq_o3_blue.^2)./(meep_freq_o3_blue.^2-1i*meep_delta_o_blue*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o4_blue.^2)./(meep_freq_o4_blue.^2-1i*meep_delta_o_blue*2*pi*f-(2*pi*f).^2) + (3*eps_lorentz*meep_freq_o5_blue.^2)./(meep_freq_o5_blue.^2-1i*meep_delta_o_blue*2*pi*f-(2*pi*f).^2) + (4*eps_lorentz*meep_freq_o6_blue.^2)./(meep_freq_o6_blue.^2-1i*meep_delta_o_blue*2*pi*f-(2*pi*f).^2);

f0_red_meep = [2.500000000000000, 2.352941176470588, 2.222222222222222, 2.105263157894737, 2.0, 1.904761904761905, 1.818181818181818];
gamma_red_meep = [0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983];
sigma_red_meep = [0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03];

f0_blue_meep = [1.818181818181818, 1.739130434782609, 1.666666666666667, 1.600000000000000, 1.538461538461538, 1.481481481481481, 1.428571428571429];
gamma_blue_meep = [0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983];
sigma_blue_meep = [0.03, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04];

f0_green_meep = [2.500000000000000, 2.352941176470588, 2.222222222222222, 2.105263157894737, 1.538461538461538, 1.481481481481481, 1.428571428571429];
gamma_green_meep = [0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983, 0.212353822772983];
sigma_green_meep = [0.05, 0.04, 0.03, 0.03, 0.03, 0.04, 0.05];

% frequency calculation
eps_red = eps_bkgd + (5*eps_lorentz*meep_freq_o_red.^2)./(meep_freq_o_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (4*eps_lorentz*meep_freq_o1_red.^2)./(meep_freq_o1_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o2_red.^2)./(meep_freq_o2_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o3_red.^2)./(meep_freq_o3_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o4_red.^2)./(meep_freq_o4_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o5_red.^2)./(meep_freq_o5_red.^2-1i*meep_delta_o_red.*f-(f).^2) + (3*eps_lorentz*meep_freq_o6_red.^2)./(meep_freq_o6_red.^2-1i*meep_delta_o_red.*f-(f).^2);
eps_green = eps_bkgd + (5*eps_lorentz*meep_freq_o_green.^2)./(meep_freq_o_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (4*eps_lorentz*meep_freq_o1_green.^2)./(meep_freq_o1_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (3*eps_lorentz*meep_freq_o2_green.^2)./(meep_freq_o2_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (3*eps_lorentz*meep_freq_o3_green.^2)./(meep_freq_o3_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (3*eps_lorentz*meep_freq_o4_green.^2)./(meep_freq_o4_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (4*eps_lorentz*meep_freq_o5_green.^2)./(meep_freq_o5_green.^2-1i*meep_delta_o_green.*f-(f).^2) + (5*eps_lorentz*meep_freq_o6_green.^2)./(meep_freq_o6_green.^2-1i*meep_delta_o_green.*f-(f).^2);
eps_blue = eps_bkgd + (3*eps_lorentz*meep_freq_o_blue.^2)./(meep_freq_o_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o1_blue.^2)./(meep_freq_o1_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o2_blue.^2)./(meep_freq_o2_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (4*eps_lorentz*meep_freq_o3_blue.^2)./(meep_freq_o3_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o4_blue.^2)./(meep_freq_o4_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (3*eps_lorentz*meep_freq_o5_blue.^2)./(meep_freq_o5_blue.^2-1i*meep_delta_o_blue.*f-(f).^2) + (4*eps_lorentz*meep_freq_o6_blue.^2)./(meep_freq_o6_blue.^2-1i*meep_delta_o_blue.*f-(f).^2);

eps_red_meep = 2.0 + (sigma_red_meep(1)*f0_red_meep(1)^2)./(f0_red_meep(1)^2 - f.^2 - 1i*f.*gamma_red_meep(1));
eps_red_meep = eps_red_meep + (sigma_red_meep(2)*f0_red_meep(2)^2)./(f0_red_meep(2)^2 - f.^2 - 1i*f.*gamma_red_meep(2));
eps_red_meep = eps_red_meep + (sigma_red_meep(3)*f0_red_meep(3)^2)./(f0_red_meep(3)^2 - f.^2 - 1i*f.*gamma_red_meep(3));
eps_red_meep = eps_red_meep + (sigma_red_meep(4)*f0_red_meep(4)^2)./(f0_red_meep(4)^2 - f.^2 - 1i*f.*gamma_red_meep(4));
eps_red_meep = eps_red_meep + (sigma_red_meep(5)*f0_red_meep(5)^2)./(f0_red_meep(5)^2 - f.^2 - 1i*f.*gamma_red_meep(5));
eps_red_meep = eps_red_meep + (sigma_red_meep(6)*f0_red_meep(6)^2)./(f0_red_meep(6)^2 - f.^2 - 1i*f.*gamma_red_meep(6));
eps_red_meep = eps_red_meep + (sigma_red_meep(7)*f0_red_meep(7)^2)./(f0_red_meep(7)^2 - f.^2 - 1i*f.*gamma_red_meep(7));

eps_blue_meep = 2.0 + (sigma_blue_meep(1)*f0_blue_meep(1)^2)./(f0_blue_meep(1)^2 - f.^2 - 1i*f.*gamma_blue_meep(1));
eps_blue_meep = eps_blue_meep + (sigma_blue_meep(2)*f0_blue_meep(2)^2)./(f0_blue_meep(2)^2 - f.^2 - 1i*f.*gamma_blue_meep(2));
eps_blue_meep = eps_blue_meep + (sigma_blue_meep(3)*f0_blue_meep(3)^2)./(f0_blue_meep(3)^2 - f.^2 - 1i*f.*gamma_blue_meep(3));
eps_blue_meep = eps_blue_meep + (sigma_blue_meep(4)*f0_blue_meep(4)^2)./(f0_blue_meep(4)^2 - f.^2 - 1i*f.*gamma_blue_meep(4));
eps_blue_meep = eps_blue_meep + (sigma_blue_meep(5)*f0_blue_meep(5)^2)./(f0_blue_meep(5)^2 - f.^2 - 1i*f.*gamma_blue_meep(5));
eps_blue_meep = eps_blue_meep + (sigma_blue_meep(6)*f0_blue_meep(6)^2)./(f0_blue_meep(6)^2 - f.^2 - 1i*f.*gamma_blue_meep(6));
eps_blue_meep = eps_blue_meep + (sigma_blue_meep(7)*f0_blue_meep(7)^2)./(f0_blue_meep(7)^2 - f.^2 - 1i*f.*gamma_blue_meep(7));

eps_green_meep = 2.0 + (sigma_green_meep(1)*f0_green_meep(1)^2)./(f0_green_meep(1)^2 - f.^2 - 1i*f.*gamma_green_meep(1));
eps_green_meep = eps_green_meep + (sigma_green_meep(2)*f0_green_meep(2)^2)./(f0_green_meep(2)^2 - f.^2 - 1i*f.*gamma_green_meep(2));
eps_green_meep = eps_green_meep + (sigma_green_meep(3)*f0_green_meep(3)^2)./(f0_green_meep(3)^2 - f.^2 - 1i*f.*gamma_green_meep(3));
eps_green_meep = eps_green_meep + (sigma_green_meep(4)*f0_green_meep(4)^2)./(f0_green_meep(4)^2 - f.^2 - 1i*f.*gamma_green_meep(4));
eps_green_meep = eps_green_meep + (sigma_green_meep(5)*f0_green_meep(5)^2)./(f0_green_meep(5)^2 - f.^2 - 1i*f.*gamma_green_meep(5));
eps_green_meep = eps_green_meep + (sigma_green_meep(6)*f0_green_meep(6)^2)./(f0_green_meep(6)^2 - f.^2 - 1i*f.*gamma_green_meep(6));
eps_green_meep = eps_green_meep + (sigma_green_meep(7)*f0_green_meep(7)^2)./(f0_green_meep(7)^2 - f.^2 - 1i*f.*gamma_green_meep(7));


%% Lumerical

wvl = linspace(0.4e-6, 0.7e-6, 50);
sz = size(wvl);
Lumerical_f = c./wvl;

Lumerical_eps_red = eps_bkgd + (5*eps_lorentz*omega_o_red.^2)./(omega_o_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o1_red.^2)./(omega_o1_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o2_red.^2)./(omega_o2_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o3_red.^2)./(omega_o3_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o4_red.^2)./(omega_o4_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o5_red.^2)./(omega_o5_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o6_red.^2)./(omega_o6_red.^2-2i*delta_o_red*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2);
Lumerical_eps_green = eps_bkgd + (5*eps_lorentz*omega_o_green.^2)./(omega_o_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o1_green.^2)./(omega_o1_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o2_green.^2)./(omega_o2_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o3_green.^2)./(omega_o3_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o4_green.^2)./(omega_o4_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o5_green.^2)./(omega_o5_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (5*eps_lorentz*omega_o6_green.^2)./(omega_o6_green.^2-2i*delta_o_green*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2);
Lumerical_eps_blue = eps_bkgd + (3*eps_lorentz*omega_o_blue.^2)./(omega_o_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o1_blue.^2)./(omega_o1_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o2_blue.^2)./(omega_o2_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o3_blue.^2)./(omega_o3_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o4_blue.^2)./(omega_o4_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (3*eps_lorentz*omega_o5_blue.^2)./(omega_o5_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2) + (4*eps_lorentz*omega_o6_blue.^2)./(omega_o6_blue.^2-2i*delta_o_blue*2*pi*Lumerical_f-(2*pi*Lumerical_f).^2);

%% file extract
filename_meep_real_red = "Red_color-filter_real_eps_meep_unit.txt";
resultm = [f;real(eps_red)];
writematrix(resultm.',filename_meep_real_red);

filename_lumerical_real_red = "Red_color-filter_real_eps_lumerical.txt";
resultm = [Lumerical_f;real(Lumerical_eps_red)];
writematrix(resultm.',filename_lumerical_real_red);

filename_meep_imag_red = "Red_color-filter_imag_eps_meep_unit.txt";
resultm = [f;imag(eps_red)];
writematrix(resultm.',filename_meep_imag_red);

filename_lumerical_imag_red = "Red_color-filter_imag_eps_lumerical.txt";
resultm = [Lumerical_f;imag(Lumerical_eps_red)];
writematrix(resultm.',filename_lumerical_imag_red);

filename_meep_real_green = "Green_color-filter_real_eps_meep_unit.txt";
resultm = [f;real(eps_green)];
writematrix(resultm.',filename_meep_real_green);

filename_lumerical_real_green = "Green_color-filter_real_eps_lumerical.txt";
resultm = [Lumerical_f;real(Lumerical_eps_green)];
writematrix(resultm.',filename_lumerical_real_green);

filename_meep_imag_green = "Green_color-filter_imag_eps_meep_unit.txt";
resultm = [f;imag(eps_green)];
writematrix(resultm.',filename_meep_imag_green);

filename_lumerical_imag_green = "Green_color-filter_imag_eps_lumerical.txt";
resultm = [Lumerical_f;imag(Lumerical_eps_green)];
writematrix(resultm.',filename_lumerical_imag_green);

filename_meep_real_blue = "Blue_color-filter_real_eps_meep_unit.txt";
resultm = [f;real(eps_blue)];
writematrix(resultm.',filename_meep_real_blue);

filename_lumerical_real_blue = "Blue_color-filter_real_eps_lumerical.txt";
resultm = [Lumerical_f;real(Lumerical_eps_blue)];
writematrix(resultm.',filename_lumerical_real_blue);

filename_meep_imag_blue = "Blue_color-filter_imag_eps_meep_unit.txt";
resultm = [f;imag(eps_blue)];
writematrix(resultm.',filename_meep_imag_blue);

filename_lumerical_imag_blue = "Blue_color-filter_imag_eps_lumerical.txt";
resultm = [Lumerical_f;imag(Lumerical_eps_blue)];
writematrix(resultm.',filename_lumerical_imag_blue);

%% Plot 

figure(1)
if option == 1

    plot(wvl.*10^9, real(Lumerical_eps_red),'k-','LineWidth', 1); % c./f.*10^(9)
    hold on;
    plot(lambda*1000, real(eps_red),'ro','LineWidth', 1);
    title('Red color-filter')
    hold on;
    xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
    ylabel('Re(eps)','fontname','arial','fontsize',16);
    legend('Lumerical','Meep','location','southeast');
    set(gca,'fontname','arial','fontsize',14)
    set(gca,'linewidth',1);
    legend boxoff

else

    plot(wvl.*10^9, imag(Lumerical_eps_red),'k-','LineWidth', 1); % c./f.*10^(9)
    hold on;
    plot(lambda*1000, imag(eps_red),'ro','LineWidth', 1);
    title('Red color-filter')
    hold on;
    xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
    ylabel('Im(eps)','fontname','arial','fontsize',16);
    legend('Lumerical','Meep','location','northeast');
    set(gca,'fontname','arial','fontsize',14)
    set(gca,'linewidth',1);
    legend boxoff

end

figure(2)
if option == 1

    plot(wvl.*10^9, real(Lumerical_eps_green),'k-','LineWidth', 1); % c./f.*10^(9)
    hold on;
    plot(lambda*1000, real(eps_green),'go','LineWidth', 1);
    title('Green color-filter')
    hold on;
    xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
    ylabel('Re(eps)','fontname','arial','fontsize',16);
    legend('Lumerical','Meep','location','southeast');
    set(gca,'fontname','arial','fontsize',14)
    set(gca,'linewidth',1);
    legend boxoff

else

    plot(wvl.*10^9, imag(Lumerical_eps_green),'k-','LineWidth', 1); % c./f.*10^(9)
    hold on;
    plot(lambda*1000, imag(eps_green),'go','LineWidth', 1);
    title('Green color-filter')
    hold on;
    xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
    ylabel('Im(eps)','fontname','arial','fontsize',16);
    legend('Lumerical','Meep','location','northeast');
    set(gca,'fontname','arial','fontsize',14)
    set(gca,'linewidth',1);
    legend boxoff

end

figure(3)
if option == 1

    plot(wvl.*10^9, real(Lumerical_eps_blue),'k-','LineWidth', 1); % c./f.*10^(9)
    hold on;
    plot(lambda*1000, real(eps_blue),'bo','LineWidth', 1);
    title('Blue color-filter')
    hold on;
    xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
    ylabel('Re(eps)','fontname','arial','fontsize',16);
    legend('Lumerical','Meep','location','southeast');
    set(gca,'fontname','arial','fontsize',14)
    set(gca,'linewidth',1);
    legend boxoff

else

    plot(wvl.*10^9, imag(Lumerical_eps_blue),'k-','LineWidth', 1); % c./f.*10^(9)
    hold on;
    plot(lambda*1000, imag(eps_blue),'bo','LineWidth', 1);
    title('Blue color-filter')
    hold on;
    xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
    ylabel('Im(eps)','fontname','arial','fontsize',16);
    legend('Lumerical','Meep','location','southeast');
    set(gca,'fontname','arial','fontsize',14)
    set(gca,'linewidth',1);
    legend boxoff

end
