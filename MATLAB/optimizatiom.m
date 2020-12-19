

close all
clear all
clc

%% fminsearch 

options = [];
fun = 'func';
x0 = [0.025 0.025 0.0017];
[x,fval,exitflag,output] = fminsearch(fun,x0,options);


%% pattersearch 

if 1==1
options = [];
x0 = [0.025 0.025 0.0017];
A = [];
b = [];
Aeq = [];
beq = [];
lb = [0.25 0.25 0.001];
ub = [0.5 0.5 0.01];
nonlcon = [];

[x1,fval1,exitflag1,output1] = patternsearch(@func,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
end

%% PSwarm

nvars = 3;
lb = [0.25 0.25 0.001];
ub = [0.5 0.5 0.01];
options = [];

[x2,fval2,exitflag2,output2] = particleswarm(@func,nvars,lb,ub,options);

x
x1
x2






