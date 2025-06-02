% 데이터 로드
f = dlmread('oled-flux.dat', '', 1, 0); % 1행(헤더) 건너뛰고 데이터 로드

% 총 방출 파워 계산
total_power = f(:,2) + f(:,3) + f(:,4) + f(:,5) + f(:,6) + f(:,7);

% 유리층과 웨이브가이드 모드 파워
glass_power = f(:,8);
waveguide_power = f(:,9) + f(:,10) + f(:,11) + f(:,12);

% 음수 플럭스 경고
if any(min(f(:,2:12), [], 'all') < 0)
    warning("Warning: flux is negative");
end

% 각 영역의 총 파워 비율
glass = glass_power ./ total_power;
waveguide = waveguide_power ./ total_power;
aluminum = 1 - glass - waveguide;

% 파장 계산 (um)
lambdas = 1 ./ f(:,1);

% 알루미늄 비율 검사
if any(aluminum < 0)
    warning("Warning: aluminum absorption is negative");
end

if any(aluminum > 1)
    warning("Warning: aluminum absorption is larger than 1");
end

% 선형 보간
lambdas_linear = linspace(0.4, 0.8, 100).';
glass_linear = interp1(lambdas, glass, lambdas_linear, "spline", "extrap");
waveguide_linear = interp1(lambdas, waveguide, lambdas_linear, "spline", "extrap");
aluminum_linear = interp1(lambdas, aluminum, lambdas_linear, "spline", "extrap");

% 플롯
figure;
hold on;
plot(lambdas_linear, glass_linear, 'b-', 'LineWidth', 2);
plot(lambdas_linear, aluminum_linear, 'r-', 'LineWidth', 2);
plot(lambdas_linear, waveguide_linear, 'g-', 'LineWidth', 2);
hold off;

xlabel("Wavelength (µm)");
ylabel("Fraction of Total Power");
legend("Glass", "Aluminum", "Organic + ITO", 'Location', 'northwest');
axis([0.4 0.8 0 1]);
grid on;

% 평균 및 표준 편차 출력
disp("Power in each region averaged over all wavelengths:");
disp(sprintf("Glass: %0.6f ± %0.6f", mean(glass_linear), std(glass_linear, 1)));
disp(sprintf("Aluminum: %0.6f ± %0.6f", mean(aluminum_linear), std(aluminum_linear, 1)));
disp(sprintf("Organic + ITO: %0.6f ± %0.6f", mean(waveguide_linear), std(waveguide_linear, 1)));
