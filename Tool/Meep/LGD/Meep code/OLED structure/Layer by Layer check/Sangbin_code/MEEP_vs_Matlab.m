close all;
clear;
clc;
% addpath('C:\Users\Haejun\Desktop\4_research\97_matlab_util_from git\materials')  
% addpath('C:\Users\Haejun\Desktop\4_research\97_matlab_util_from git\mie_spheres')  
% addpath('C:\Users\Haejun\Desktop\4_research\97_matlab_util_from git')  
const = MATLAB_CONSTANTS;

R_TFSF = load('Reflection_my_code.txt'); %meep fdtd result
T_TFSF = load('Transmission_my_code.txt');

% data_reflectance = load('RefractiveIndex_SiO2.txt');
% data_transmittance = load('Refractiveindex_SiO2_transmittance(400-700nm).txt');
% 
% SiN_refractiveIndex = load('RefractiveIndex_SiN(400-700nm).txt');

% data_eps_mat=load('Lorentz_red_eps.txt');
% data_eps_mat=load('Lorentz_green_eps.txt');
% data_eps_mat=load('Lorentz_blue_eps.txt');

% data_real_eps_mat=load('Red_color-filter_real_eps_lumerical.txt');
% data_imag_eps_mat=load('Red_color-filter_imag_eps_lumerical.txt');
% data_real_eps_mat = load('red_eps_re.txt');
% data_imag_eps_mat = load('red_eps_im.txt');
% data_real_eps_mat=load('Green_color-filter_real_eps_lumerical.txt');
% data_imag_eps_mat=load('Green_color-filter_imag_eps_lumerical.txt');
% data_real_eps_mat = load('green_eps_re.txt');
% data_imag_eps_mat = load('green_eps_im.txt');
% data_real_eps_mat=load('Blue_color-filter_real_eps_lumerical.txt');
% data_imag_eps_mat=load('Blue_color-filter_imag_eps_lumerical.txt');
% data_real_eps_mat = load('blue_eps_re.txt');
% data_imag_eps_mat = load('blue_eps_im.txt');
% red_data_real_eps_meep = load('Red_color-filter_real_eps_meep_unit.txt');
% red_data_imag_eps_meep = load('Red_color-filter_imag_eps_meep_unit.txt');
% green_data_real_eps_meep = load('Green_color-filter_real_eps_meep_unit.txt');
% green_data_imag_eps_meep = load('Green_color-filter_imag_eps_meep_unit.txt');
% blue_data_real_eps_meep = load('Blue_color-filter_real_eps_meep_unit.txt');
% blue_data_imag_eps_meep = load('Blue_color-filter_imag_eps_meep_unit.txt');

% data_eps_mat=load('SiO2_eps_re.txt');
% data_eps_mat=flipud(data_eps_mat);

% data_eps_mat = load('Si_eps.txt');
% 
% data_eps_re=load('Si_eps_re.txt');
% data_eps_im=load('Si_eps_im.txt');
% data_eps_re = flipud(data_eps_re); % Si data is fliped
% data_eps_im = flipud(data_eps_im);

% cSi_data_real = load('Crystalline_Silicon_real_eps_meep_unit.txt');
% cSi_data_imag = load('Crystalline_Silicon_imag_eps_meep_unit.txt');

Nw = length(T_TFSF);
% Nw = 64; % 64 -> SiO2, 67 -> cSi, 69 -> Green, 68 -> Red, Blue
Nw_meep = 100;
Nw_cSi = 1000;

data_freq = linspace(1.0/(0.40), 1.0/(0.70), Nw).';
data_lambda =1./data_freq;

wvl = linspace(0.40, 0.70, Nw);
wvl = wvl.';
f = 1.0./wvl;

n_air = 1.0;

%% Al2O3

f0_Al2O3_meep = [10.45201873779817, 10.520778320800689];
gamma_Al2O3_meep = [0.0, 0.0, 0.0]; % it is already devided by 2pi
sigma_Al2O3_meep = [0.7485390758646482, 1.321424754906942];

eps_Al2O3_meep = 1 + (sigma_Al2O3_meep(1)*f0_Al2O3_meep(1)^2)./(f0_Al2O3_meep(1)^2 - f.^2 - 1i*f.*gamma_Al2O3_meep(1));
eps_Al2O3_meep = eps_Al2O3_meep + (sigma_Al2O3_meep(2)*f0_Al2O3_meep(2)^2)./(f0_Al2O3_meep(2)^2 - f.^2 - 1i*f.*gamma_Al2O3_meep(2));




% n_SiO2 = zeros(1,length(data_reflectance));
% n_SiN = zeros(1,length(SiN_refractiveIndex));
% R0 = zeros(1,length(data_reflectance));
% X = zeros(1,length(data_reflectance));
% T0 = zeros(1,length(data_transmittance));
% Y = zeros(1,length(data_transmittance));

%% SiO2

% for m = 1 : length(data_reflectance)
%     X(m) = data_reflectance(m,1);
%     n_SiO2(m) = data_reflectance(m,2);
%     R0(m) = (abs((n_air - n_SiO2(m))./(n_air + n_SiO2(m)))).^2;
% end
% 
% for k = 1 : length(data_transmittance)
%     Y(k) = data_transmittance(k,1);
%     T0(k) = data_transmittance(k,2);
% end

%% SiN

% f0_sin_meep = 1.0/0.190891752117013;
% gamma_sin_meep = 1.0/3.11518072864322;
% sigma_sin_meep = 1.2650;
% 
% eps_sin_meep = 2.320 + (sigma_sin_meep*f0_sin_meep^2)./(f0_sin_meep^2 - f.^2 - 1i*f.*gamma_sin_meep);

% for n = 1 : length(SiN_refractiveIndex)
%     n_SiN(n) = SiN_refractiveIndex(n,2);
%     R0(n) = (abs((n_air - n_SiN(n))./(n_air + n_SiN(n)))).^2;
%     T0(n) = 1.0 - R0(n);
% end

%% Silica(SiO2)

% f0_silica_meep = [1.0/0.0684043, 1.0/0.1162414, 1.0/9.896161];
% gamma_silica_meep = [0.0, 0.0, 0.0]; % it is already devided by 2pi
% sigma_silica_meep = [0.696166300, 0.407942600, 0.897479400];
% 
% eps_silica_meep = 1.0 + (sigma_silica_meep(1)*f0_silica_meep(1)^2)./(f0_silica_meep(1)^2 - f.^2 - 1i*f.*gamma_silica_meep(1));
% eps_silica_meep = eps_silica_meep + (sigma_silica_meep(2)*f0_silica_meep(2)^2)./(f0_silica_meep(2)^2 - f.^2 - 1i*f.*gamma_silica_meep(2));
% eps_silica_meep = eps_silica_meep + (sigma_silica_meep(3)*f0_silica_meep(3)^2)./(f0_silica_meep(3)^2 - f.^2 - 1i*f.*gamma_silica_meep(3));

%% TMM

Vacuum =@(w) (1.00)*ones(size(w));
%a = 500e-9;
a= 1.0;
layerThicknesses = [0.3 * a];
pol = {'s'}; %p=TM, s= TE
theta = 0;
w = 2 * pi .*f;

for k = 1 : Nw % 610 : 673(1 : 64) -> SiO2(540~560nm) 
%     color_filter_eps = @(w)(data_eps_mat(k,2) + 1.0i*data_eps_mat(k,3) -0.001*1i); % -0.001*1i
%     color_filter_eps = @(w)(data_real_eps_mat(k,2) + 1.0i*(data_imag_eps_mat(k,2) - 0.001) ); % -0.001*1i
%     color_filter_eps = @(w)(data_real_eps_mat(Nw-k+1,2) + 1.0i*(data_imag_eps_mat(Nw-k+1,2) - 0.001) ); % -0.001*1i
%     color_filter_eps = @(w)(data_real_eps_mat(932+k,2) + 1.0i*(data_imag_eps_mat(932+k,2) - 0.001) ); % 466~535(540-560nm), 933~1000(680-700nm)
%     color_filter_eps_meep = @(w)(red_data_real_eps_meep(k,2) + 1.0i*red_data_imag_eps_meep(k,2));
%     color_filter_eps_meep = @(w)(green_data_real_eps_meep(k,2) + 1.0i*green_data_imag_eps_meep(k,2));
%     color_filter_eps_meep = @(w)(blue_data_real_eps_meep(k,2) + 1.0i*blue_data_imag_eps_meep(k,2));

%     PEC_eps = @(w) ( 1.0 + 10000.0*1.0i);
%     SiO2_eps = @(w)(data_eps_mat(609+k,2));
%     SiO2_eps = @(w)(eps_silica_meep(k));
%     boundary_eps = @(w)(1.5 + 0.0*1i);eps_SiO2_meep 
%     SiN_eps = @(w)(real(eps_sin_meep(k)) + imag(eps_sin_meep(k))*1.0i);
    Al2O3_eps = @(w)(real(eps_Al2O3_meep (k)) + imag(eps_Al2O3_meep (k))*1.0i);
%     Si_eps = @(w)(data_eps_re(644+k,2) + 1i*data_eps_im(644+k,2)); % 540-560nm range
%     Si_eps = @(w)(data_eps_re(k,2) + 1i*data_eps_im(k,2));

    layerMaterials = {Vacuum, Al2O3_eps,  Vacuum};
%     layerMaterials_lumerical = {Vacuum, SiN_eps, SiN_eps};
%     layerMaterials_meep = {Vacuum, color_filter_eps_meep, Vacuum};

%     layerMaterials = {Vacuum, SiO2_eps, SiO2_eps};
%     layerMaterials = {Vacuum, SiN_eps, Vacuum};
%     layerMaterials = {Vacuum, Si_eps, Si_eps};
%     layerMaterials = {Vacuum, PEC_eps, Vacuum};    

    [R,T,A,r,t] = multilayer_film(layerMaterials, layerThicknesses, w, theta, pol);
%     [R_lumerical,T_lumerical,A_lumerical,r_lumerical,t_lumerical] = multilayer_film(layerMaterials_lumerical, layerThicknesses, w, theta, pol);
%     [R_meep,T_meep,A_meep,r_meep,t_meep] = multilayer_film(layerMaterials_meep, layerThicknesses, w, theta, pol);

    my_T(k) = T{1}(k);      
    my_R(k) = R{1}(k);

%     lumerical_T(k) = T_lumerical{1}(k);
%     lumerical_R(k) = R_lumerical{1}(k);
% 
%     meep_T(k) = T_meep{1}(k);
%     meep_R(k) = R_meep{1}(k);
    
%     FDTD_T(k) = data_trn_mat(k); %/data_free(k);
%     FDTD_R(k) = data_ref_mat(k);
end

for k = 1 : Nw_meep
%     FDTD_T(Nw_meep-k+1) = data_trn_mat(k)/data_free(k);
%     FDTD_R(Nw_meep-k+1) = data_ref_mat(k); %/data_free(k);
%     FDTD_T_my_code(Nw_meep-k+1) = data_trn_mat_my_code(k)/data_free_my_code(k);
%     FDTD_R_my_code(Nw_meep-k+1) = data_ref_mat_my_code(k)/data_free_my_code(k);
    FDTD_T_my_code(Nw_meep-k+1) = T_TFSF(k); %/data_free_my_code(k);
    FDTD_R_my_code(Nw_meep-k+1) = R_TFSF(k); %/data_free_my_code(k);
end

% figure(1)
% % plot(data_lambda*1000, data_eps_mat(:,2), 'ko', 'linewidth', 1);
% % plot(data_lambda*1000, data_real_eps_mat(:,2), 'ko', 'linewidth', 1);
% plot(data_lambda*1000, data_eps_re(:,2), 'ko', 'linewidth', 1);
% hold on
% % plot(data_lambda*1000, real(red_data_real_eps_meep(:,2)), 'r-', 'linewidth', 2);
% % plot(data_lambda*1000, real(green_data_real_eps_meep(:,2)), 'g-', 'linewidth', 2);
% % plot(data_lambda*1000, real(blue_data_real_eps_meep(:,2)), 'b-', 'linewidth', 2);
% plot(data_lambda_cSi*1000, cSi_data_real(:,2), 'c-', 'linewidth', 2);
% title('Red color-filter');
% % title('Green color-filter');
% % title('Blue color-filter');
% % title('c-Si');
% xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
% ylabel('Re(eps)','fontname','arial','fontsize',16);
% legend('Lumerical','Meep','location','southeast');
% set(gca,'fontname','arial','fontsize',14)
% set(gca,'linewidth',1);
% legend boxoff
% 
% figure(2)
% % plot(data_lambda*1000, data_eps_mat(:,3), 'ko', 'linewidth', 1);
% % plot(data_lambda*1000, data_imag_eps_mat(:,2), 'ko', 'linewidth', 1);
% plot(data_lambda*1000, data_eps_im(:,2), 'ko', 'linewidth', 1);
% hold on
% % plot(data_lambda*1000, red_data_imag_eps_meep(:,2), 'r-', 'linewidth', 2);
% % plot(data_lambda*1000, green_data_imag_eps_meep(:,2), 'g-', 'linewidth', 2);
% % plot(data_lambda*1000, blue_data_imag_eps_meep(:,2), 'b-', 'linewidth', 2);
% plot(data_lambda_cSi*1000, cSi_data_imag(:,2), 'c-', 'linewidth', 2);
% title('Red color-filter');
% % title('Green color-filter');
% % title('Blue color-filter');
% % title('c-Si');
% xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
% ylabel('Im(eps)','fontname','arial','fontsize',16);
% legend('Lumerical','Meep','location','northeast');
% set(gca,'fontname','arial','fontsize',14)
% set(gca,'linewidth',1);
% legend boxoff

figure(3)
%plot(lambda./1e-9, angle(r{1}) / pi + 1.0, 'r-', 'linewidth', 2);
% plot(data_lambda_meep*1000, FDTD_T, 'ko', 'linewidth', 1);
% plot(data_lambda, meep_T, 'ko', 'linewidth', 1);
plot(wvl*1000, my_T, 'k-', 'linewidth', 2);
% plot(data_lambda*1000, lumerical_T, 'b-','linewidth', 2);
hold on
% plot(data_lambda*1000, my_T, 'c-', 'linewidth', 2);
plot(data_lambda*1000, FDTD_T_my_code, 'bo', 'LineWidth',1);
% plot(Y*1000, T0, 'k--', 'LineWidth',2);
% plot(SiN_refractiveIndex(:,1)*1000, T0, 'b--', 'LineWidth', 1);
% plot(data_lambda_cSi*1000, cSi_T, 'c-', 'linewidth', 2);
% hold on
% plot(data_lambda*1000, meep_T, 'go', 'linewidth', 1);
% hold on
% plot(data_lambda*1000, lumerical_T, 'b-','linewidth', 2);
% plot(data_lambda, FDTD_R, 'r-','linewidth', 2);
% ylim([0 1.0]);
%xlim([min(lambda./1e-9) max(lambda./1e-9)]);
% title('SiO2(res:30)');
title('Al2O3');
% title('silica')
% title('c-Si(res:40)');
% title('a-Si');
% title('Red color-filter(res:50)');
% title('Green color-filter(res:50)');
% title('Blue color-filter(res:50)');
% title('PEC(res:30)');
xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
ylabel('Transmission','fontname','arial','fontsize',16);
% legend('Meep transmission','Meep unit conversion','location','northeast');
legend('TMM','FDTD', 'location','southeast'); % 'Analytic(Transmittance)'
% legend('TMM(Lumerical)','FDTD(my code)', 'Analytic(Transmittance)','location','southeast'); % '
% legend('FDTD(Meep validation code)','TMM(Lumerical)', 'FDTD(my code)','location','southeast');
set(gca,'fontname','arial','fontsize',14)
set(gca,'linewidth',1);
legend boxoff


figure(4)
%plot(lambda./1e-9, angle(r{1}) / pi + 1.0, 'r-', 'linewidth', 2);
% plot(data_lambda_meep*1000, FDTD_R, 'ko', 'linewidth', 1);
plot(wvl*1000, my_R, 'k-', 'linewidth', 2);
% plot(data_lambda, meep_R, 'ko', 'linewidth', 1);
% plot(data_lambda*1000, lumerical_R, 'r-','linewidth', 2);
hold on
% plot(data_lambda*1000, my_R, 'g-', 'linewidth', 2);
% plot(data_lambda_cSi*1000, cSi_R, 'c-', 'linewidth', 2);
% plot(data_lambda, meep_R, 'g-', 'linewidth', 2);
plot(data_lambda*1000, FDTD_R_my_code, 'bo', 'LineWidth',1);
% plot(X*1000, R0, 'ko', 'LineWidth',1);
% plot(SiN_refractiveIndex(:,1)*1000, R0, 'b--', 'LineWidth',1);
% hold on
% plot(data_lambda*1000, lumerical_R, 'b-','linewidth', 2);
% plot(data_lambda, FDTD_R, 'r-','linewidth', 2);
% xlim([400 700])
% ylim([0 1]);
%xlim([min(lambda./1e-9) max(lambda./1e-9)]);
% title('SiO2(res:30)');
title('Al2O3');
% title('silica')
% title('c-Si(res:40)');
% title('a-Si');
% title('Red color-filter(res:30)');
% title('Green color-filter(res:50)');
% title('Blue color-filter(res:50)');
% title('PEC(res:30)');
xlabel('Wavelength (nm)','fontname','arial','fontsize',16);
ylabel('Reflection','fontname','arial','fontsize',16);
% legend('Meep Reflection','Meep unit conversion','location','northeast');
% legend('Meep Reflection','Lumerical Reflection(Matlab)','location','northeast');
% legend('FDTD(Meep)','TMM(Lumerical)','location','northeast');
% legend('FDTD(Meep validation code)','FDTD(my code)','location','northeast');
% legend('TMM(Lumerical)','FDTD(my code)', 'Analytic(Reflectance)','location','northeast');
legend('TMM','FDTD','location','northeast');
set(gca,'fontname','arial','fontsize',14)
set(gca,'linewidth',1);
legend boxoff