
% beta is 1 x Ntheta
% n, t are Nw x 1
% D is Nw x Ntheta x 4 {D11, D12, D21, D22}

% w, beta normalized by c/w
% t normalized by w/c
function[D,Dinv,P,kx] = LayerMatrices(Nw, n, beta, sp, t)

Nt = length(beta);
if (length(n)>1)
    n = repmat(n,1,Nt);
else
    n = repmat(n,Nw,Nt);
end
beta = repmat(beta,Nw,1);
kx = sqrt(n.^2 - beta.^2);
I = ones(Nw,Nt);

D = zeros(Nw,Nt,4);
Dinv = zeros(Nw,Nt,4);
P = zeros(Nw,Nt,4);

if (strcmp(sp,'s')) % s wave
    D(:,:,1) = I;
    D(:,:,2) = I;
    D(:,:,3) = kx;
    D(:,:,4) = -kx;
    Dinv(:,:,1) = 0.5*I;
    Dinv(:,:,2) = 0.5./kx;
    Dinv(:,:,3) = 0.5*I;
    Dinv(:,:,4) = -0.5./kx;
else % p wave
    D(:,:,1) = kx ./ n;
    D(:,:,2) = kx ./ n;
    D(:,:,3) = n;
    D(:,:,4) = -n;
    Dinv(:,:,1) = 0.5*n./kx;
    Dinv(:,:,2) = 0.5./n;
    Dinv(:,:,3) = 0.5*n./kx;
    Dinv(:,:,4) = -0.5./n;
end

if (nargin==5)
    t = repmat(t,1,Nt);
    P(:,:,1) = exp(-1i*kx.*t);
    P(:,:,2) = zeros(Nw,Nt,1);
    P(:,:,3) = zeros(Nw,Nt,1);
    P(:,:,4) = exp(1i*kx.*t);
end

end