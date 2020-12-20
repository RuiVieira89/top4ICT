


close all
clear all
clc



%% fminsearch 

options = [];

lb = [];
ub = [];
x0 = [0 0]; 

fun = 'func';
[x,fval,exitflag,output] = fminsearch(fun,x0,options);


%% pattersearch 

options = [];
A = [];
b = [];
Aeq = [];
beq = [];
nonlcon = [];

[x1,fval1,exitflag1,output1] = patternsearch(@func,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);


%% PSwarm

nvars = 2;
options = [];

[x2,fval2,exitflag2,output2] = particleswarm(@func,nvars,lb,ub,options);

%% PLOTS

[X,Y] = meshgrid(-2:.1:2);     

Z(length(X), length(Y)) = 0;

for i=1:length(X)
    for j=1:length(Y)
        Z(i,j) = func([X(i), Y(j)]);
    end
end

surf(X,Y,Z)
hold on
scatter3(x2(1), x2(2),fval2+5, 500,'.', 'r')

shading interp

print = '\n%s X=[%.4f, %.4f]; fval=%.4f \n';
fprintf(print, 'fminsearch', x(1), x(2), fval)
fprintf(print, 'pattersearch', x1(1), x1(2), fval1)
fprintf(print, 'PSwarm', x2(1), x2(2), fval2)







