
% This code computes the reflection and transmission properties of a 
% film with arbitrarily many layers.
% Premise: two half-spaces with N layers in between (thus: N+2 materials)
%   e.g. Air-SiO2-Si-SiO2-Air would be three layers, surrounded by two air
%   half-spaces
% NOTE: layer_thickness * frequency needs to be dimensionless
% Constraints: 
%    -- non-evanescent waves
%    -- initial/final layers assumed to be non-absorbing
% Implementation follows Yeh, Optical Waves in Layered Media, sec. 5.1

function[R,T,A,r,t] = multilayer_film(layerMaterials, layerThicknesses, w, theta, pol)

N = length(layerMaterials);
if (N ~= length(layerThicknesses)+2)
    error('Need to specify N layer thicknesses and N+2 materials');
end
% incident and exit half-spaces
hs1Material = layerMaterials{1};
hs2Material = layerMaterials{end};

Nw = length(w);
beta = sin(theta); % normalized by c/w

R = cell(2,1); T = cell(2,1); A = cell(2,1);
r = cell(2,1); t = cell(2,1);
for p = 1:length(pol)
    [Ds,~,~,kxs] = LayerMatrices(Nw,sqrt(hs2Material(w)), beta, pol(p));
    M = Ds;
    for i = N-1:-1:2
        [Dl,Dlinv,Pl] = LayerMatrices(Nw,sqrt(layerMaterials{i}(w)), beta, pol(p), layerThicknesses(i-1)*w);
        M = MTimes(Dl,MTimes(Pl,MTimes(Dlinv,M)));
    end
    [~,D0inv,~,kx0] = LayerMatrices(Nw,sqrt(hs1Material(w)), beta, pol(p));
    M = MTimes(D0inv,M);
    
    M11 = M(:,:,1);
    M21 = M(:,:,3);
    r{p} = M21 ./ M11;
    t{p} = 1 ./ M11;

    R{p} = abs(r{p}).^2;
    T{p} = kxs./kx0 .* abs(t{p}).^2;
    A{p} = 1 - R{p} - T{p};
end

end
