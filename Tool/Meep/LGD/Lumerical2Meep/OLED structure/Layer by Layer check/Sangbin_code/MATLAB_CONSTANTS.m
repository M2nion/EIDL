
function[const] = MATLAB_CONSTANTS

const = struct();
const.c = 299792458;
const.q = 1.602176565e-19;
const.e0 = 8.85418782e-12;
const.mu0 = pi*4e-7;
const.k = 1.3806488e-23; % J/K
const.h = 6.62606957e-34;
const.hbar = const.h / 2 / pi;
const.wToEV = const.hbar / const.q;
const.Z0 = sqrt(const.mu0 / const.e0);
const.me = 9.10938356e-31;
const.alpha = const.q^2 * const.Z0 / (2 * const.h); 
% should have unit conversions here

end