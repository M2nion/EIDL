clear all;
clc
close all;

pol = 1;           % polarization (0 = TE, 1 = TM)
Nf_dft=1;    % Number of Dft frequancy

custom_red = importdata('custom_red.mat');
custom_green = importdata('custom_green.mat');
custom_blue = importdata('custom_blue.mat');
fields_data = 'dft-fields.h5';
hinfo_fields = h5info('dft-fields.h5');
flux_data = 'dft-flux-blue.h5';
hinfo_flux = h5info('dft-flux-blue.h5');


switch pol


    case 0
        Ez = 'dft-fields_red.h5';
        Hx = 'dft-fields_red.h5';
        Hy = 'dft-fields_red.h5';

              % Hx
        Name_Hxr1=["/hx_0.r"];
        Name_Hxi1=["/hx_0.i"];    
        Hx_data_r1 = h5read(Hx,Name_Hxr1);          %Real
        Hx_data_i1 = h5read(Hx,Name_Hxi1);           %Imagine
        Hx_data1 = Hx_data_r1 + Hx_data_i1 * 1.0i;       %Complex
        % Hy
        Name_Hyr1=["/hy_0.r"];
        Name_Hyi1=["/hy_0.i"];
        Hy_data_r1 = h5read(Hy,Name_Hyr1);          %Real
        Hy_data_i1 = h5read(Hy,Name_Hyi1);           %Imagine
        Hy_data1 = Hy_data_r1 + Hy_data_i1 * 1.0i;       %Complex
        % Ez
        Name_Ezr1=["/ez_0.r"];
        Name_Ezi1=["/ez_0.i"];
        Ez_data_r1 = h5read(Ez,Name_Ezr1);            %Real
        Ez_data_i1 = h5read(Ez,Name_Ezi1);             %Imagine
        Ez_data1 = Ez_data_r1 + Ez_data_i1 * 1.0i;         %Complex

        %Hx
        Hx_data_r = Hx_data_r1;      %Real
        Hx_data_i = Hx_data_i1;      %Imagine
        Hx_data = Hx_data_r+ Hx_data_i * 1.0i;      %Complex
        %Hy
        Hy_data_r = Hy_data_r1;      %Real
        Hy_data_i = Hy_data_i1;       %Imagine
        Hy_data = Hy_data_r + Hy_data_i * 1.0i;     %Complex
        %Ez
        Ez_data_r = Ez_data_r1;         %Real
        Ez_data_i = Ez_data_i1;          %Imagine
        Ez_data = Ez_data_r + Ez_data_i * 1.0i;     %Complex

        %% Calculate Poynting Vector & Phase

        sz=size(Ez_data1);% 결과데이터 행렬 크기 (Ly x Lx)
        
        Sx=zeros(sz(1,1),sz(1,2)); % S_x(x,y)
        Sy=zeros(sz(1,1),sz(1,2)); % S_y(x,y)
        Fx=zeros(sz(1,1),sz(1,2)); % S_x(x,y)
        Fy=zeros(sz(1,1),sz(1,2)); % S_y(x,y)
        MS=zeros(sz(1,1),sz(1,2));% amplitude
        SS=zeros(sz(1,1),sz(1,2)); % Phase
        FS=zeros(sz(1,1),sz(1,2)); 
        FOM=zeros(sz(1,1),sz(1,2));
        
        for k=1:sz(1,1) % 행(y)
            for m=1:sz(1,2) % 열(x)
        %% Source = Ez
               %if ((k>=3.5*rs) && (k<=7.0*rs))
                E=[0, 0, Ez_data(k, m)];                                             % E vector
                E_conj=[0, 0, conj(Ez_data(k,m))];
                H=[conj(Hx_data(k,m)), conj(Hy_data(k,m)), 0];  % H* vector
        
                P=real(cross(E, H)); % Calculating Poynting vector (x, y, z)
                F=abs(dot(E,E_conj));
            
                Sx(k,m)=P(1,1)*0.5;
                Sy(k,m)=P(1,2)*0.5;
                Fx(k,m)=F*0.5;
                Fy(k,m)=F*0.5;

                FS(k,m)=0.5*(Fx(k,m)+Fy(k,m));
                MS(k,m)=sqrt(Sx(k,m)^2+Sy(k,m)^2); % |S(x,y)|, amplitude of poynting vector
                SS(k,m)=(asin(Sy(k,m)/MS(k,m)));       % +-, ++ (-pi/2~pi/2)  phase of poynting vector
                if(Sx(k,m)<0&&Sy(k,m)<0) % - - (-pi~ -pi/2)
                    SS(k,m)=-pi-SS(k,m);
                end
                if(Sx(k,m)<0&&Sy(k,m)>0)% -+ (pi/2~pi)
                    SS(k,m)=pi-SS(k,m);
                end
             
               %end
            end
            Progress = k/(sz(1,1))
        end
        %% figure1 (Plane wave1: E_z field)
        %  
        figure
        surf(abs(Ez_data1.*conj(Ez_data1)),'EdgeColor','none','LineStyle','none','FaceLighting','phong') % took 1./eps to make the high epsilon region to be black. field
        
        hold on
        
        c = colorbar('Ticks',[0, 13000],'TickLabels',{'0', '1'});
        c.Label.String = '|E|^2';
        c.Label.Position = [0.7, 14000];
        c.Label.Rotation = 0;
        colormap(custom_red);
        caxis([0 13000])
        set(gca,'fontname','arial','fontsize',12,'fontweight','bold')
        set(gca,'linewidth',3);
        
        view(0,90);
        axis off
        grid off
        %print("Ez_r_P1_"+n,'-dpng','-r900') 
        %set(gcf, 'Color', 'None')
        
        %% figure5  (amplitude of Poynting Vector)
%         
%         figure
%         surf(MS,'EdgeColor','none','LineStyle','none','FaceLighting','phong') % took 1./eps to make the high epsilon region to be black. field
%         
%         hold on
%         %c = colorbar();
%         c = colorbar('Ticks',[0, 15000],'TickLabels',{'0', '1'});
%         c.Label.String = 'Intensity (a.u)';
%         c.Label.Position = [-0.8, 2.15];
%         c.Label.Rotation = 0;
%         caxis([0 15000])
%         colormap(hot);
%         %colormap(flipud(gray));
%         set(gca,'fontname','arial','fontsize',12,'fontweight','bold')
%         set(gca,'linewidth',3);
%         
%         view(2);% = view(0,90) < 방위각, 고도각
%         
%         grid off
%         axis off

 case 1
    Hz = 'dft-flux-blue.h5';
    Ex = 'dft-flux-blue.h5';
%     Ey = 'dft-fields-green.h5';
 for n = 0:Nf_dft -1 % Dft 주파수 개수만큼 반복시작

    %% extract planewave1 data

    % Hz
    Name_Hzr1=["/hz_0.r"];
    Name_Hzi1=["/hz_0.i"]; 
    Hz_data_r1 = h5read(Hz,Name_Hzr1);                %Real
    Hz_data_i1 = h5read(Hz,Name_Hzi1);                 %Imagine
    Hz_data1 = Hz_data_r1 + Hz_data_i1 * 1.0i;         %Complex
    % Ex
    Name_Exr1=["/ex_0.r"];
    Name_Exi1=["/ex_0.i"];
    Ex_data_r1 = h5read(Ex,Name_Exr1);                      %Real
    Ex_data_i1 = h5read(Ex,Name_Exi1);                       %Imagine
    Ex_data1 = Ex_data_r1 + Ex_data_i1 * 1.0i;                      %Complex
    % Ey
%     Name_Eyr1=["/ey_0.r"];
%     Name_Eyi1=["/ey_0.i"];
%     Ey_data_r1 = h5read(Ey,Name_Eyr1);                      %Real
%     Ey_data_i1 = h5read(Ey,Name_Eyi1);                       %Imagine 
%     Ey_data1 = Ey_data_r1 + Ey_data_i1 * 1.0i;                      %Complex

    %Hz
    Hz_data_r = Hz_data_r1;      %Real
    Hz_data_i = Hz_data_i1;      %Imagine
    Hz_data = Hz_data_r+ Hz_data_i * 1.0i;                            %Complex
    %Ex
    Ex_data_r = Ex_data_r1;      %Real
    Ex_data_i = Ex_data_i1;       %Imagine
    Ex_data = Ex_data_r + Ex_data_i * 1.0i;                           %Complex
    %Ey
%     Ey_data_r = Ey_data_r1;         %Real
%     Ey_data_i = Ey_data_i1;          %Imagine
%     Ey_data = Ey_data_r + Ey_data_i * 1.0i;                             %Complex

    %% Calculate Poynting Vector & Phase
    
    sz=size(Hz_data1);% 결과데이터 행렬 크기 (Ly x Lx)
    
    Sx=zeros(sz(1,1),sz(1,2)); % S_x(x,y)
    Sy=zeros(sz(1,1),sz(1,2)); % S_y(x,y)
    Fx=zeros(sz(1,1),sz(1,2)); % S_x(x,y)
    Fy=zeros(sz(1,1),sz(1,2)); % S_y(x,y)
    MS=zeros(sz(1,1),sz(1,2));% amplitude
    SS=zeros(sz(1,1),sz(1,2)); % Phase
    FS=zeros(sz(1,1),sz(1,2)); 
    Flux = zeros(sz(1,1),sz(1,2)); 

    FOM=0;
    for k=1:sz(1,1) % 행(y)
        for m=1:sz(1,2) % 열(x)
            %% Source = Hz 
%             E=[Ex_data(k, m), Ey_data(k, m), 0];
%             E_conj=[conj(Ex_data(k,m)), conj(Ey_data(k,m)), 0];
            E=[Ex_data(k, m), 0, 0];
            E_conj=[conj(Ex_data(k,m)), 0, 0];
            H=[0, 0, conj(Hz_data(k, m))];  % H* vector
    
            P=real(cross(E, H)); % Calculating Poynting vector (x, y, z)
            Flux(k,m)=real(Ex_data(k,m) * conj(Hz_data(k,m)));
            F=abs(E).^2;
        
            Sx(k,m)=P(1,1)*0.5;
            Sy(k,m)=-P(1,2)*0.5;
            Fx(k,m)=F(1,1)*0.5;
%             Fy(k,m)=F*0.5;

            FS(k,m)=0.5*(Fx(k,m));
            MS(k,m)=sqrt(Sx(k,m)^2+Sy(k,m)^2); % |S(x,y)|, amplitude of poynting vector
            SS(k,m)=(asin(Sy(k,m)/MS(k,m)));       % +-, ++ (-pi/2~pi/2)  phase of poynting vector
            if(Sx(k,m)<0&&Sy(k,m)<0) % - - (-pi~ -pi/2)
                SS(k,m)=-pi-SS(k,m);
            end
            if(Sx(k,m)<0&&Sy(k,m)>0)% -+ (pi/2~pi)
                SS(k,m)=pi-SS(k,m);
            end
            
           
    
            
        end
    end
Sum_flux = 0;
Sum_crosstalk = 0;
% for i = 1:27
%     Flux_red = Flux(1,i);
%     Sum_flux = Sum_flux + Flux_red;
% end
% for i = 1:26
%     Flux_green = Flux(1,i+27);
%     Sum_crosstalk = Sum_crosstalk + Flux_green;
% end
% for i=1:26
%     Flux_blue = Flux(1,i+53);
%     Sum_crosstalk = Sum_crosstalk + Flux_blue;
% end
% for i = 1:27
%     Flux_green2 = Flux(1,i+79);
%     Sum_crosstalk = Sum_crosstalk + Flux_green2;
% end

%% figure1 (Plane wave: E_z field)
%  
figure
surf(MS,'EdgeColor','none','LineStyle','none','FaceLighting','phong') % took 1./eps to make the high epsilon region to be black. field

hold on

c = colorbar('Ticks',[0 7000],'TickLabels',{'0', '1'});
c.Label.String = 'Flux';
c.Label.Position = [0.7, 7700];
c.Label.Rotation = 0;
colormap(custom_red);
caxis([0 7000])
set(gca,'fontname','arial','fontsize',12,'fontweight','bold')
set(gca,'linewidth',3);

view(0,90);
axis off
grid off
%print("Ez_r_P1_"+n,'-dpng','-r900') 
%set(gcf, 'Color', 'None')

%% figure5  (amplitude of Poynting Vector)

figure
surf(FS,'EdgeColor','none','LineStyle','none','FaceLighting','phong') % took 1./eps to make the high epsilon region to be black. field

hold on
%c = colorbar();
c = colorbar('Ticks', [0 1000],'TickLabels',{'0', '1'});
c.Label.String = '|E|^2';
c.Label.Position = [0.7, 1100];
c.Label.Rotation = 0;
caxis([0 1000])
colormap("hot");
%colormap(flipud(gray));
set(gca,'fontname','arial','fontsize',12,'fontweight','bold')
set(gca,'linewidth',3);

view(2);% = view(0,90) < 방위각, 고도각

grid off
axis off

%% figure (dir of Poynting vector)



 end
end
