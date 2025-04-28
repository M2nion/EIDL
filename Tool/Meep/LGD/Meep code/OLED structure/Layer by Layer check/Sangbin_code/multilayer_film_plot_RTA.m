
function[] = multilayer_film_plot_RTA(lambda, theta, R, T, A, pol)

if (length(theta)==1)
    for p = 1:length(pol)
        figure;
        plot(lambda, [R{p} T{p} A{p}]);
        xlim([min(lambda) max(lambda)]);
        titleStr = sprintf('%s polarization, angle = %g degrees', pol{p}, theta);
        title(titleStr);
        xlabel('Wavelength (nm)');
        ylabel('Refl / Trans / Abs');
        legend({'R','T','A'});
    end
else
    for p = 1:length(pol)
        figure;
        imagesc(theta, lambda, R{p});
        titleStr = sprintf('Reflection, %s Polarization', pol{p});
        title(titleStr);
        xlabel('Angle (degrees)')
        ylabel('Wavelength (nm)');
        
        figure;
        imagesc(theta, lambda, T{p});
        titleStr = sprintf('Transmission, %s Polarization', pol{p});
        title(titleStr);
        xlabel('Angle (degrees)')
        ylabel('Wavelength (nm)');
        
        figure;
        imagesc(theta, lambda, A{p});
        titleStr = sprintf('Absorption, %s Polarization', pol{p});
        title(titleStr);
        xlabel('Angle (degrees)')
        ylabel('Wavelength (nm)');
    end
end
