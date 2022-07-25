%%%%%%%%%%%%%%%% 
close all
clear 
clc
mode=1; % 1 for VFSA, 2 for SPSA, 
funcy=3; %%%% 1 for sign function 2D with global minimum at(0,0) where f(x, y)=0.0
%%%%%%%%%%%%% 2 for Banana(Rosen)function with global minimum at(1,1) where f(x, y)=0.0
%%%%%%%%%%%%% 3 for Griewank function with global minimum at(0,0) where f(x, y)=0.0
reproduce=1; % 0 for random, 1 for reproducable results
nruns=1; %%%%% Number of runs(cases) 
n=50; % number of iteration per run
xx=zeros(n,nruns); %%% tracer for x position at different iteration 
yy=zeros(n,nruns); %%% tracer for y position at different iteration 
err_plot=zeros(n,nruns); %%% tracer for error at different iteration per run
xx_best=zeros(nruns,1); yy_best=zeros(nruns,1); 
error_tresh=0.001;
%%%% range of two varaibles for optimization
xmax=10; xmin=-10; dx=0.05;
ymax=10; ymin=-10; dy=0.05;
nx=round(abs(xmax-xmin)./dx);
ny=round(abs(ymax-ymin)./dy);

if mode==2; %%% Parameter definition for basic SPSA
    divider=100;%%% to manage boundary conditions    
    p=2; %%% number of variables to be optimized
    alpha =0.602; %%%% alpha =1;
    gamma =0.101; %%%% gamma =1/6;
    a=300; %%% default 0.0017; note that by small a SPSA will...
           %%be a local optimizaer and large a make it a global optimizer
    c=1.9;% chosen by standard guidelines default 1.9
    A=5;%%%%% default 50 or  A=0.1*n;
    theta=zeros(1,p);
    theta_min=[xmin,ymin];   %lower bounds on theta  
    theta_max=[xmax,ymax];  %upper bounds on theta
    dtheta=[dx,dy];
    ntheta=[nx,ny];
elseif mode==1; %%%% Parameter definition for basic VFSA
    nmov=3;
    temp0=1;
    decay=0.999;
    t0(1)=1;
    t0(2)=1;   
else
    sprintf('mode must be either 1 or 2')
    return
end

if reproduce==1;
    rand('seed',31415927)
    randn('seed',3111113)
end


%%%% Main loop
for jrun=1:nruns;     
    if mode==2; %%%%% SPSA
        for jtheta=1:p; %% loop over model parameters
            theta(jtheta)=monte_carlo_sample(theta_min(jtheta),dtheta(jtheta),ntheta(jtheta),theta_max(jtheta));%%% Initial guess
        end
        k=1;
        xx(k,jrun)=theta(1);
        yy(k,jrun)=theta(2);  
        err_plot(k,jrun)=(1-func1(theta(1),theta(2))).^2;
        while(k<=n-1);
            ak=a/(k+A)^alpha;
            ck=c/(k)^gamma;
            delta= 2*round(rand(1,p))-1; %%% Bernoulli Dist. +1 and -1
            thetaplus=theta+ck*delta;
            thetaminus=theta-ck*delta;     
            yplus=(1-func1(thetaplus(1),thetaplus(2))).^2;
            yminus=(1-func1(thetaminus(1),thetaminus(2))).^2;
            ghat=(yplus-yminus)./(2*ck*delta);
            theta=theta-ak*ghat;
            for jtheta=1:p; %% loop over model parameters
                if ((theta(jtheta)>=theta_max(jtheta)));
                    theta(jtheta)=theta_max(jtheta)-(theta_max(jtheta)-theta_min(jtheta))/divider;
                elseif ((theta(jtheta)<=theta_min(jtheta)));
                    theta(jtheta)=theta_min(jtheta)+(theta_max(jtheta)-theta_min(jtheta))/divider;
                end          
            end          
            err_plot(k+1,jrun)=(1-func1(theta(1),theta(2))).^2; 
            xx(k+1,jrun)=theta(1);
            yy(k+1,jrun)=theta(2);
            %%%%%%%%%%%%% Exit from k loop if error is so small
            if err_plot(k,jrun)<=error_tresh;
                k=n;                
            end            
            k=k+1;       
        end
        xx_best(jrun)=theta(1);
        yy_best(jrun)=theta(2);        
        
    else %%% VFSA
        xmod=monte_carlo_sample(xmin,dx,nx,xmax);%%% Initial guess
        ymod=monte_carlo_sample(ymin,dy,ny,ymax);
        emod=(1-func1(xmod,ymod)).^2;
        jtemp=1;
        xx(jtemp,jrun)=xmod;
        yy(jtemp,jrun)=ymod;  
        err_plot(jtemp,jrun)=funcy_gen2d(xmod,ymod,funcy);
        while(jtemp<=n-1);
            temp(jtemp)=temp0.*exp(-decay.*(jtemp-1).^0.5);
            tmp1=t0(1).*exp(-decay.*(jtemp-1).^0.5); tmp2=t0(2).*exp(-decay.*(jtemp-1).^0.5);
            for jmov=1:nmov;
                xtrial=walk(xmod,dx,xmin,xmax,tmp1);
                ytrial=walk(ymod,dy,ymin,ymax,tmp2);    
                xtrial=round((xtrial-xmin)./dx).*dx+xmin;
                ytrial=round((ytrial-ymin)./dy).*dy+ymin;
                etrial=funcy_gen2d(xtrial,ytrial,funcy);
                if etrial< emod;
                    %%%% hist_updat
                    emod=etrial;
                    xmod=xtrial;
                    ymod=ytrial;
                else
                    arg=(etrial-emod)./temp(jtemp);
                    if arg>1.e6;
                        pde=0.001;
                    else
                        pde=exp(-arg);
                    end
                    if pde>rand;
                        %%%% hist_updat
                        emod=etrial;
                        xmod=xtrial;
                        ymod=ytrial;
                    end
                end
            end %%%% end move
            err_plot(jtemp+1,jrun)=emod;    
            xx(jtemp+1,jrun)=xmod;
            yy(jtemp+1,jrun)=ymod;
            %%%%%%%%%%%%% Exit from temp loop if emod is so small
            if emod<=error_tresh;
                jtemp=n;                
            end            
            jtemp=jtemp+1;       
        end
        xx_best(jrun)=xmod;
        yy_best(jrun)=ymod;        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%plotting
lw=2;
fs=12;
%%%% range of two varaibles for plotting only 
xmax=10; xmin=-10; dx=0.05;
ymax=10; ymin=-10; dy=0.05;
nx=round(abs(xmax-xmin)./dx);
ny=round(abs(ymax-ymin)./dy);
x=xmin:dx:xmax;
y=ymin:dy:ymax;
%%%%% generate function to display
myfun=zeros(ny,nx);
for i=1:nx;
    for j=1:ny;
        myfun(j,i)=funcy_gen2d(x(i),y(j),funcy);
    end 
end

figure(1)
imagesc(x,y,myfun);
colorbar;xlabel('X Axes','FontSize',fs);ylabel('Y Axes','FontSize',fs);set(gca,'FontSize',fs)
title('Objective Function','FontSize',fs);
for jrun=1:nruns
    hold on
    plot(xx(:,jrun),yy(:,jrun),'--w','LineWidth',lw); %%% all iteration
    hold on
    plot(xx(1,jrun),yy(1,jrun),'dw'); %%% first iteration for jrun(initial guess)
    hold on
    plot(xx(n,jrun),yy(n,jrun),'*k');%%% last iteration for jrun(solution)
    colorbar;xlabel('X Axes','FontSize',fs);ylabel('Y Axes','FontSize',fs);set(gca,'FontSize',fs)
end


figure(2)
plot(err_plot,'r','LineWidth',lw)
xlabel('Iteration number','FontSize',fs);
ylabel('Objective function','FontSize',fs);
set(gca,'FontSize',fs)

figure(3)        
    subplot(1,2,1)
    plot(xx,'b','LineWidth',lw)
    xlabel('Iteration number','FontSize',fs);
    ylabel('First model parameter(x)','FontSize',fs);
    ylim([xmin,xmax]);
    grid;
    set(gca,'FontSize',fs)
    subplot(1,2,2)
    plot(yy,'b','LineWidth',lw)
    xlabel('Iteration number','FontSize',fs);
    ylabel('Second model parameter(y)','FontSize',fs);
    ylim([ymin,ymax]);
    grid;
    set(gca,'FontSize',fs)

if nruns>1
    figure(4)
    subplot(3,1,1)
    imagesc(err_plot');colorbar;
    ylabel('Number of runs','FontSize',fs);
    title('Objective Function','FontSize',fs);
    subplot(3,1,2)
    imagesc(xx');colorbar;
    ylabel('Number of runs','FontSize',fs);
    title('First model parameter(x)','FontSize',fs);
    subplot(3,1,3)
    imagesc(yy');colorbar;
    xlabel('Iteration number','FontSize',fs);
    ylabel('Number of runs','FontSize',fs);
    title('Second model parameter(y)','FontSize',fs);
end




        
    
    
    
    
    
    
    
    