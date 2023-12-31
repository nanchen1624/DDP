clear;
clc;
% use strategy generated by dual value to calculate the lower bound
% This is the stock acquisition from Andew Lo and D. Bertsimas (1998)
% We use the linear price impact with information as the example

%%  problem description
t1=clock;
T=20; % number of periods
X1=0;
R1=100;
NR=3;
NX=2;
A=[30,7,3;7,25,-5;3,-5,20];
E=20*eye(NR);
B=[50,20;30,20;10,40];
C=[0.8,0.1;0.2,0.6];
sigma_eta=[10,2;2,8];

%% problem setting
K=3*10^4;
iteration=6;
% state
X=zeros(iteration+1,NX,T,K);%information
R=zeros(iteration+1,NR,T,K);% remaining stock
R(:,:,1,:)=ones(iteration+1,NR,K)*R1;
%regression cofficient
F=zeros(iteration+1,T);%1
G1=zeros(iteration+1,NX,T);%X_t
G2=zeros(iteration+1,NR,T);%R_t
H1=zeros(iteration+1,NX,NX,T);%X_t^2
H2=zeros(iteration+1,NR,NR,T);%R_t^2
H3=zeros(iteration+1,NX,NR,T);%X_tR_t
K1=zeros(iteration+1,NX,NR,T);%X_tR_t^2
K2=zeros(iteration+1,NX,NR,T);%X_t^2R_t^2
K3=zeros(iteration+1,NX,T);%X_t^4
K33=zeros(iteration+1,T);%X_t^2X_t^2
K4=zeros(iteration+1,NR,NR,T);%R_t^2R^2
K5=zeros(iteration+1,NR,NR,T);%R_t*R_t^0.5
K6=zeros(iteration+1,NX,T);%X_t^3
K7=zeros(iteration+1,NR,T);%R_t^3
K8=zeros(iteration+1,NX,NR,T);%X_t^2R_t
for i=1:iteration+1
    H2(i,:,:,T)=A;
    H3(i,:,:,T)=B';
    K5(i,:,:,T)=E;
end

J=zeros(iteration+1,T,K);
W=zeros(iteration+1,1);
W_var=zeros(iteration+1,1);

% generate state
Eta=zeros(iteration+1,NX,T,K);
for i=1:iteration+1
    for t=1:T-1
        Eta(i,:,t+1,:)=mvnrnd(zeros(1,NX),sigma_eta,K)';
        X(i,:,t+1,:)=C*squeeze(X(i,:,t,:))+squeeze(Eta(i,:,t+1,:));
        %S_temp=R1/T*2*rand(K,1);
        %R(i,t+1,:)=max(squeeze(R(i,t,:))-S_temp,0);
        R(i,:,t+1,:)=R1*rand(NR,K);
    end
end

%% initial state from the fitted value iteration approximation
u = zeros(NR, T, K);
options = optimset('Algorithm','sqp','Display','off');
i=1
for t = T-1:-1:1
    for n=1:K
        [u(:,t,n),J(i,t,n)]=fmincon(@(S) objfun_low_hd_nl(S,A,B,E,C,squeeze(R(i,:,t,n))',...
                squeeze(X(i,:,t,n))',squeeze(G2(i,:,t+1))',squeeze(H3(i,:,:,t+1)),squeeze(H2(i,:,:,t+1)),...
                squeeze(K1(i,:,:,t+1)),squeeze(K2(i,:,:,t+1)),squeeze(K5(i,:,:,t+1)),squeeze(K4(i,:,:,t+1))',diag(sigma_eta),squeeze(K7(i,:,t+1))',squeeze(K8(i,:,:,t+1))),...
                R(i,:,t,n)'/(T-t+1),[-eye(NR);eye(NR)],[zeros(NR,1);squeeze(R(i,:,t,n))'],...
                    [],[],zeros(NR,1),squeeze(R(i,:,t,n))',[],options);
         J(i,t,n) = J(i,t,n) + F(i,t+1) + squeeze(G1(i,:,t+1)) * C * squeeze(X(i,:,t,n))' + (C*squeeze(X(i,:,t,n))')'* squeeze(H1(i,:,:,t+1))*C*squeeze(X(i,:,t,n))'+sum(sum(sigma_eta.* squeeze(H1(i,:,:,t+1))))+...
             squeeze(K3(i,:,t+1))*((C*squeeze(X(i,:,t,n))').^4+6*(C*squeeze(X(i,:,t,n))').^2.*diag(sigma_eta)+ 3 *diag(sigma_eta))+...
             squeeze(K6(i,:,t+1))*((C*squeeze(X(i,:,t,n))').^3+3*C*squeeze(X(i,:,t,n))'.*diag(sigma_eta));
    end

    if t>1
        M_temp=[ones(K,1),squeeze(X(i,1,t,:)),squeeze(X(i,2,t,:)),squeeze(R(i,1,t,:)),squeeze(R(i,2,t,:)),squeeze(R(i,3,t,:)),...
        squeeze(X(i,1,t,:).^2),squeeze(X(i,2,t,:).^2),squeeze(X(i,1,t,:).*X(i,2,t,:)),squeeze(R(i,1,t,:).^2),squeeze(R(i,2,t,:).^2),squeeze(R(i,3,t,:).^2),...
        squeeze(R(i,1,t,:).*R(i,2,t,:)),squeeze(R(i,1,t,:).*R(i,3,t,:)),squeeze(R(i,2,t,:).*R(i,3,t,:)),...
        squeeze(R(i,1,t,:).*X(i,1,t,:)),squeeze(R(i,2,t,:).*X(i,1,t,:)),squeeze(R(i,3,t,:).*X(i,1,t,:)),...
        squeeze(R(i,1,t,:).*X(i,2,t,:)),squeeze(R(i,2,t,:).*X(i,2,t,:)),squeeze(R(i,3,t,:).*X(i,2,t,:)),...
        squeeze(R(i,1,t,:).^2.*X(i,1,t,:)),squeeze(R(i,2,t,:).^2.*X(i,1,t,:)),squeeze(R(i,3,t,:).^2.*X(i,1,t,:)),...
        squeeze(R(i,1,t,:).^2.*X(i,2,t,:)),squeeze(R(i,2,t,:).^2.*X(i,2,t,:)),squeeze(R(i,3,t,:).^2.*X(i,2,t,:)),...
        squeeze(R(i,1,t,:).*X(i,1,t,:)).^2,squeeze(R(i,2,t,:).*X(i,1,t,:)).^2,squeeze(R(i,3,t,:).*X(i,1,t,:)).^2,...
        squeeze(R(i,1,t,:).*X(i,2,t,:)).^2,squeeze(R(i,2,t,:).*X(i,2,t,:)).^2,squeeze(R(i,3,t,:).*X(i,2,t,:)).^2,...
        squeeze(X(i,1,t,:)).^4,squeeze(X(i,2,t,:)).^4,squeeze(R(i,1,t,:)).^4,squeeze(R(i,2,t,:)).^4,squeeze(R(i,3,t,:)).^4,...
        squeeze(R(i,1,t,:).^1.5),squeeze(R(i,2,t,:).^1.5),squeeze(R(i,3,t,:).^1.5),...
        squeeze(X(i,1,t,:)).^3,squeeze(X(i,2,t,:)).^3,squeeze(R(i,1,t,:)).^3,squeeze(R(i,2,t,:)).^3,squeeze(R(i,3,t,:)).^3,...
        squeeze(R(i,1,t,:).*R(i,2,t,:)).^2,squeeze(R(i,1,t,:).*R(i,3,t,:)).^2,squeeze(R(i,2,t,:).*R(i,3,t,:)).^2];
        Coff_temp=regress(squeeze(J(i,t,:)),M_temp);
        F(i,t)=Coff_temp(1);
        G1(i,1,t)=Coff_temp(2);
        G1(i,2,t)=Coff_temp(3);
        G2(i,1,t)=Coff_temp(4);
        G2(i,2,t)=Coff_temp(5);
        G2(i,3,t)=Coff_temp(6);
        H1(i,1,1,t)=Coff_temp(7);
        H1(i,2,2,t)=Coff_temp(8);
        H1(i,1,2,t)=Coff_temp(9)/2;
        H1(i,2,1,t)=Coff_temp(9)/2;
        H2(i,1,1,t)=Coff_temp(10);
        H2(i,2,2,t)=Coff_temp(11);
        H2(i,3,3,t)=Coff_temp(12);
        H2(i,1,2,t)=Coff_temp(13)/2;
        H2(i,1,3,t)=Coff_temp(14)/2;
        H2(i,2,3,t)=Coff_temp(15)/2;
        H2(i,2,1,t)=Coff_temp(13)/2;
        H2(i,3,1,t)=Coff_temp(14)/2;
        H2(i,3,2,t)=Coff_temp(15)/2;
        H3(i,1,1,t)=Coff_temp(16);
        H3(i,1,2,t)=Coff_temp(17);
        H3(i,1,3,t)=Coff_temp(18);
        H3(i,2,1,t)=Coff_temp(19);
        H3(i,2,2,t)=Coff_temp(20);
        H3(i,2,3,t)=Coff_temp(21);
        K1(i,1,1,t)=Coff_temp(22);
        K1(i,1,2,t)=Coff_temp(23);
        K1(i,1,3,t)=Coff_temp(24);
        K1(i,2,1,t)=Coff_temp(25);
        K1(i,2,2,t)=Coff_temp(26);
        K1(i,2,3,t)=Coff_temp(27);
        K2(i,1,1,t)=Coff_temp(28);
        K2(i,1,2,t)=Coff_temp(29);
        K2(i,1,3,t)=Coff_temp(30);
        K2(i,2,1,t)=Coff_temp(31);
        K2(i,2,2,t)=Coff_temp(32);
        K2(i,2,3,t)=Coff_temp(33);
        K3(i,1,t)=Coff_temp(34);
        K3(i,2,t)=Coff_temp(35);
        K4(i,1,1,t)=Coff_temp(36);
        K4(i,2,2,t)=Coff_temp(37);
        K4(i,3,3,t)=Coff_temp(38);
        K5(i,1,1,t)=Coff_temp(39);
        K5(i,2,2,t)=Coff_temp(40);
        K5(i,3,3,t)=Coff_temp(41);

        K6(i,1,t)=Coff_temp(42);
        K6(i,2,t)=Coff_temp(43);
        K7(i,1,t)=Coff_temp(44);
        K7(i,2,t)=Coff_temp(45);
        K7(i,3,t)=Coff_temp(46);

        K4(i,1,2,t)=Coff_temp(47)/2;
        K4(i,1,3,t)=Coff_temp(48)/2;
        K4(i,2,3,t)=Coff_temp(49)/2;
        K4(i,2,1,t)=Coff_temp(47)/2;
        K4(i,3,1,t)=Coff_temp(48)/2;
        K4(i,3,2,t)=Coff_temp(49)/2;
     else
        W(i)=mean(J(i,t,:))
        W_var(i)=var(J(i,t,:))
    end
end

%% problem solving, get dual value using DDP
for i=2:iteration+1
    for t=1:(T-1)
        % intitial guess
        RS_temp=ones(T-t+1)-triu(ones(T-t+1));  
        RS_temp=kron(RS_temp,eye(NR));
        H3_temp=squeeze(H3(i-1,:,:,t+1));
        G1_temp=reshape(squeeze(G1(i-1,:,t+1:T)),(T-t)*NX,1);
        H1_sum=sum(squeeze(H1(i-1,:,:,t+1:T)),3);
        H1_exp=sum(sum(H1_sum.*sigma_eta));
        H1_temp=squeeze(H1(i-1,:,:,t+1));
        K1_temp=squeeze(K1(i-1,:,:,t+1));
        K2_temp=squeeze(K2(i-1,:,:,t+1));
        K8_temp=squeeze(K8(i-1,:,:,t+1));
        K3_temp=reshape(squeeze(K3(i-1,:,t+1:T)),(T-t)*NX,1);
        K6_temp=reshape(squeeze(K6(i-1,:,t+1:T)),(T-t)*NX,1);
        for s=t+2:T
            H1_temp=mdiag(H1_temp,squeeze(H1(i-1,:,:,s)));
            H3_temp=mdiag(H3_temp,squeeze(H3(i-1,:,:,s)));
            K1_temp=mdiag(K1_temp,squeeze(K1(i-1,:,:,s)));
            K2_temp=mdiag(K2_temp,squeeze(K2(i-1,:,:,s)));
            K8_temp=mdiag(K8_temp,squeeze(K8(i-1,:,:,s)));
        end
        H3_temp=[zeros((T-t)*NX,NR),H3_temp];
        K1_temp=[zeros((T-t)*NX,NR),K1_temp];
        K2_temp=[zeros((T-t)*NX,NR),K2_temp];
        K8_temp=[zeros((T-t)*NX,NR),K8_temp];
        A_temp=kron(eye(T-t+1),A);
        C_temp=kron(eye(T-t),C);
        B_temp=kron(eye(T-t+1),B);
        sigma_temp=kron(ones(T-t,1),diag(sigma_eta));
        sigma_temp_in=kron(eye(T-t),[sigma_eta(2,2),sigma_eta(1,1)]);
        E_temp=kron(eye(T-t+1),E);
        K33_temp=kron(diag(K33(i-1,t+1:T)),[0,1;0,0]);
        
        parfor k=1:K
           X_temp=reshape(squeeze(X(i,:,t:T,k)),(T-t+1)*NX,1);
           X_temp_1=reshape(squeeze(X(i,1,t+1:T,k)),(T-t),1);
           X_temp_2=reshape(squeeze(X(i,2,t+1:T,k)),(T-t),1);
           Eta_temp=reshape(squeeze(Eta(i,:,t+1:T,k)),(T-t)*NX,1);
           R_t=kron(ones(T-t+1,1),squeeze(R(i,:,t,k))');
           [fu,fvalue]=fmincon(@(S) objfun_hd_nl(S,X_temp,R_t,A_temp,B_temp,E_temp,RS_temp,Eta_temp,H3_temp,NX,K1_temp,K2_temp,K8_temp,C_temp,sigma_temp),R1/T*ones((T-t+1)*NR,1),...
                -eye((T-t+1)*NR),zeros((T-t+1)*NR,1),kron(ones(1,T-t+1),diag(ones(NR,1))),squeeze(R(i,:,t,k))',zeros(NR*(T-t+1),1),kron(ones(T-t+1,1),squeeze(R(i,:,t,k))'),[],options);
            J(i,t,k)=fvalue-G1_temp'*Eta_temp+H1_exp-Eta_temp'*H1_temp*Eta_temp-2*(C_temp*X_temp(1:end-NX))'*H1_temp*Eta_temp+...
                K3_temp'*(-4*(C_temp*X_temp(1:end-NX)).^3.*Eta_temp+6*(C_temp*X_temp(1:end-NX)).^2.*(sigma_temp-Eta_temp.^2)-4*(C_temp*X_temp(1:end-NX)).*(Eta_temp.^3)+...
                3*sigma_temp.^2-Eta_temp.^4)+K6_temp'*(-3*(C_temp*X_temp(1:end-NX)).^2.*Eta_temp+3*(C_temp*X_temp(1:end-NX)).*(sigma_temp-Eta_temp.^2)-Eta_temp.^3)+...
                ((C_temp*X_temp(1:end-NX)).^2)'*K33_temp*((C_temp*X_temp(1:end-NX)).^2)+K33(i-1,t+1:T)*sigma_temp_in*((C_temp*X_temp(1:end-NX)).^2)+(2*(T-t)*sigma_eta(1,2)^2+(T-t)*sigma_eta(1,1)*sigma_eta(2,2))*sum(K33(i-1,t+1:T))-...
                K33(i-1,t+1:T)*(X_temp_1.^2.*X_temp_2.^2);
        end
        if t>1
            % update parameters after first iteration
            M_temp=[ones(K,1),squeeze(X(i,1,t,:)),squeeze(X(i,2,t,:)),squeeze(R(i,1,t,:)),squeeze(R(i,2,t,:)),squeeze(R(i,3,t,:)),...
            squeeze(X(i,1,t,:).^2),squeeze(X(i,2,t,:).^2),squeeze(X(i,1,t,:).*X(i,2,t,:)),squeeze(R(i,1,t,:).^2),squeeze(R(i,2,t,:).^2),squeeze(R(i,3,t,:).^2),...
            squeeze(R(i,1,t,:).*R(i,2,t,:)),squeeze(R(i,1,t,:).*R(i,3,t,:)),squeeze(R(i,2,t,:).*R(i,3,t,:)),...
            squeeze(R(i,1,t,:).*X(i,1,t,:)),squeeze(R(i,2,t,:).*X(i,1,t,:)),squeeze(R(i,3,t,:).*X(i,1,t,:)),...
            squeeze(R(i,1,t,:).*X(i,2,t,:)),squeeze(R(i,2,t,:).*X(i,2,t,:)),squeeze(R(i,3,t,:).*X(i,2,t,:)),...
            squeeze(R(i,1,t,:).^2.*X(i,1,t,:)),squeeze(R(i,2,t,:).^2.*X(i,1,t,:)),squeeze(R(i,3,t,:).^2.*X(i,1,t,:)),...
            squeeze(R(i,1,t,:).^2.*X(i,2,t,:)),squeeze(R(i,2,t,:).^2.*X(i,2,t,:)),squeeze(R(i,3,t,:).^2.*X(i,2,t,:)),...
            squeeze(R(i,1,t,:).*X(i,1,t,:)).^2,squeeze(R(i,2,t,:).*X(i,1,t,:)).^2,squeeze(R(i,3,t,:).*X(i,1,t,:)).^2,...
            squeeze(R(i,1,t,:).*X(i,2,t,:)).^2,squeeze(R(i,2,t,:).*X(i,2,t,:)).^2,squeeze(R(i,3,t,:).*X(i,2,t,:)).^2,...
            squeeze(X(i,1,t,:)).^4,squeeze(X(i,2,t,:)).^4,squeeze(R(i,1,t,:)).^4,squeeze(R(i,2,t,:)).^4,squeeze(R(i,3,t,:)).^4,...
            squeeze(R(i,1,t,:).^1.5),squeeze(R(i,2,t,:).^1.5),squeeze(R(i,3,t,:).^1.5),...
            squeeze(X(i,1,t,:)).^3,squeeze(X(i,2,t,:)).^3,squeeze(R(i,1,t,:)).^3,squeeze(R(i,2,t,:)).^3,squeeze(R(i,3,t,:)).^3,...
            squeeze(R(i,1,t,:).*R(i,2,t,:)).^2,squeeze(R(i,1,t,:).*R(i,3,t,:)).^2,squeeze(R(i,2,t,:).*R(i,3,t,:)).^2];
            Coff_temp=regress(squeeze(J(i,t,:)),M_temp);
            F(i,t)=Coff_temp(1);
            G1(i,1,t)=Coff_temp(2);
            G1(i,2,t)=Coff_temp(3);
            G2(i,1,t)=Coff_temp(4);
            G2(i,2,t)=Coff_temp(5);
            G2(i,3,t)=Coff_temp(6);
            H1(i,1,1,t)=Coff_temp(7);
            H1(i,2,2,t)=Coff_temp(8);
            H1(i,1,2,t)=Coff_temp(9)/2;
            H1(i,2,1,t)=Coff_temp(9)/2;
            H2(i,1,1,t)=Coff_temp(10);
            H2(i,2,2,t)=Coff_temp(11);
            H2(i,3,3,t)=Coff_temp(12);
            H2(i,1,2,t)=Coff_temp(13)/2;
            H2(i,1,3,t)=Coff_temp(14)/2;
            H2(i,2,3,t)=Coff_temp(15)/2;
            H2(i,2,1,t)=Coff_temp(13)/2;
            H2(i,3,1,t)=Coff_temp(14)/2;
            H2(i,3,2,t)=Coff_temp(15)/2;
            H3(i,1,1,t)=Coff_temp(16);
            H3(i,1,2,t)=Coff_temp(17);
            H3(i,1,3,t)=Coff_temp(18);
            H3(i,2,1,t)=Coff_temp(19);
            H3(i,2,2,t)=Coff_temp(20);
            H3(i,2,3,t)=Coff_temp(21);
            K1(i,1,1,t)=Coff_temp(22);
            K1(i,1,2,t)=Coff_temp(23);
            K1(i,1,3,t)=Coff_temp(24);
            K1(i,2,1,t)=Coff_temp(25);
            K1(i,2,2,t)=Coff_temp(26);
            K1(i,2,3,t)=Coff_temp(27);
            K2(i,1,1,t)=Coff_temp(28);
            K2(i,1,2,t)=Coff_temp(29);
            K2(i,1,3,t)=Coff_temp(30);
            K2(i,2,1,t)=Coff_temp(31);
            K2(i,2,2,t)=Coff_temp(32);
            K2(i,2,3,t)=Coff_temp(33);
            K3(i,1,t)=Coff_temp(34);
            K3(i,2,t)=Coff_temp(35);
            K4(i,1,1,t)=Coff_temp(36);
            K4(i,2,2,t)=Coff_temp(37);
            K4(i,3,3,t)=Coff_temp(38);
            K5(i,1,1,t)=Coff_temp(39);
            K5(i,2,2,t)=Coff_temp(40);
            K5(i,3,3,t)=Coff_temp(41);

            K6(i,1,t)=Coff_temp(42);
            K6(i,2,t)=Coff_temp(43);
            K7(i,1,t)=Coff_temp(44);
            K7(i,2,t)=Coff_temp(45);
            K7(i,3,t)=Coff_temp(46);
            K4(i,1,2,t)=Coff_temp(47)/2;
            K4(i,1,3,t)=Coff_temp(48)/2;
            K4(i,2,3,t)=Coff_temp(49)/2;
            K4(i,2,1,t)=Coff_temp(47)/2;
            K4(i,3,1,t)=Coff_temp(48)/2;
            K4(i,3,2,t)=Coff_temp(49)/2;
        else
            W(i)=mean(J(i,t,:))
            W_var(i)=var(J(i,t,:))
        end
    end
end

%% compute upper bound
M=10^4; % lower bound sample path
J_low=zeros(iteration+1,M);
u_low=zeros(iteration+1,NR,T,M);
W_low=zeros(iteration+1,1);
W_varlow=zeros(iteration+1,1);

% generate state
Eta_low=zeros(iteration+1,NX,T,M);
X_low=zeros(iteration+1,NX,T,M);%information
R_low=zeros(iteration+1,NR,T,M);% remaining stock
R_low(:,:,1,:)=ones(iteration+1,NR,M)*R1;
for i=1:iteration+1
    for t=1:T-1
        Eta_low(i,:,t+1,:)=mvnrnd(zeros(1,NX),sigma_eta,M)';
        X_low(i,:,t+1,:)=C*squeeze(X_low(i,:,t,:))+squeeze(Eta_low(i,:,t+1,:));
    end
end
% compute upper bound for each iteration
for i=1:iteration+1
    for t=1:T-1
        parfor n=1:M
            u_low(i,:,t,n)=fmincon(@(S) objfun_low_hd_nl(S,A,B,E,C,squeeze(R_low(i,:,t,n))',...
                squeeze(X_low(i,:,t,n))',squeeze(G2(i,:,t+1))',squeeze(H3(i,:,:,t+1)),squeeze(H2(i,:,:,t+1)),...
                squeeze(K1(i,:,:,t+1)),squeeze(K2(i,:,:,t+1)),squeeze(K5(i,:,:,t+1)),squeeze(K4(i,:,:,t+1))',diag(sigma_eta),squeeze(K7(i,:,t+1))',squeeze(K8(i,:,:,t+1))),...
                R_low(i,:,t,n)'/(T-t+1),[-eye(NR);eye(NR)],[zeros(NR,1);squeeze(R_low(i,:,t,n))'],...
                    [],[],zeros(NR,1),squeeze(R_low(i,:,t,n))',[],options);
        end
        R_low(i,:,t+1,:)=squeeze(R_low(i,:,t,:))-squeeze(u_low(i,:,t,:));
        for n=1:M
            J_low(i,n)=J_low(i,n)+u_low(i,:,t,n)*E*sqrt(abs(u_low(i,:,t,n)))'+(A*u_low(i,:,t,n)'+B*X_low(i,:,t,n)')'*R_low(i,:,t,n)';
        end
    end
    for n=1:M
        u_low(i,:,T,n)=R_low(i,:,T,n);
        J_low(i,n)=J_low(i,n)+u_low(i,:,T,n)*E*sqrt(abs(u_low(i,:,T,n)))'+(A*u_low(i,:,T,n)'+B*X_low(i,:,T,n)')'*R_low(i,:,T,n)';
    end
    W_low(i)=mean(J_low(i,:))
    W_varlow(i)=var(J_low(i,:))    
end

%% covex programming check
J_check=zeros(1,K);
u_check=zeros(NR,T,K);

% generate state
Eta_check=zeros(NX,T,K);
X_check=zeros(NX,T,K);%information
R_check=zeros(NR,T,K);% remaining stock
R_check(:,1,:)=ones(NR,K)*R1;

for t=1:T-1
    Eta_check(:,t+1,:)=mvnrnd(zeros(1,NX),sigma_eta,K)';
    X_check(:,t+1,:)=C*squeeze(X_check(:,t,:))+squeeze(Eta_check(:,t+1,:));
end

i = iteration+1
for t=1:T-1
    parfor n=1:K
        u_check(:,t,n)=fmincon(@(S) objfun_low_hd_nl(S,A,B,E,C,squeeze(R_check(:,t,n)),...
            squeeze(X_check(:,t,n)),squeeze(G2(i,:,t+1))',squeeze(H3(i,:,:,t+1)),squeeze(H2(i,:,:,t+1)),...
            squeeze(K1(i,:,:,t+1)),squeeze(K2(i,:,:,t+1)),squeeze(K5(i,:,:,t+1)),squeeze(K4(i,:,:,t+1))',diag(sigma_eta),squeeze(K7(i,:,t+1))',squeeze(K8(i,:,:,t+1))),...
            R_check(:,t,n)/(T-t+1),[-eye(NR);eye(NR)],[zeros(NR,1);squeeze(R_check(:,t,n))],...
                [],[],zeros(NR,1),squeeze(R_check(:,t,n)),[],options);
    end
    R_check(:,t+1,:)=squeeze(R_check(:,t,:))-squeeze(u_check(:,t,:));
    for n=1:M
        J_check(n)=J_check(n)+u_check(:,t,n)'*E*sqrt(abs(u_check(:,t,n)))+(A*u_check(:,t,n)+B*X_check(:,t,n))'*R_check(:,t,n);
    end
end
for n=1:M
    u_check(:,T,n)=R_check(:,T,n);
    J_check(n)=J_check(n)+u_check(:,T,n)'*E*sqrt(abs(u_check(:,T,n)))+(A*u_check(:,T,n)+B*X_check(:,T,n))'*R_check(:,T,n);
end
W_check=mean(J_check)
W_varcheck=var(J_check) 

t=1
RS_temp=ones(T-t+1)-triu(ones(T-t+1));  
RS_temp=kron(RS_temp,eye(NR));
H3_temp=squeeze(H3(i-1,:,:,t+1));
G1_temp=reshape(squeeze(G1(i-1,:,t+1:T)),(T-t)*NX,1);
H1_sum=sum(squeeze(H1(i-1,:,:,t+1:T)),3);
H1_exp=sum(sum(H1_sum.*sigma_eta));
H1_temp=squeeze(H1(i-1,:,:,t+1));
K1_temp=squeeze(K1(i-1,:,:,t+1));
K2_temp=squeeze(K2(i-1,:,:,t+1));
K8_temp=squeeze(K8(i-1,:,:,t+1));
K3_temp=reshape(squeeze(K3(i-1,:,t+1:T)),(T-t)*NX,1);
K6_temp=reshape(squeeze(K6(i-1,:,t+1:T)),(T-t)*NX,1);
for s=t+2:T
    H1_temp=mdiag(H1_temp,squeeze(H1(i-1,:,:,s)));
    H3_temp=mdiag(H3_temp,squeeze(H3(i-1,:,:,s)));
    K1_temp=mdiag(K1_temp,squeeze(K1(i-1,:,:,s)));
    K2_temp=mdiag(K2_temp,squeeze(K2(i-1,:,:,s)));
    K8_temp=mdiag(K8_temp,squeeze(K8(i-1,:,:,s)));
end
H3_temp=[zeros((T-t)*NX,NR),H3_temp];
K1_temp=[zeros((T-t)*NX,NR),K1_temp];
K2_temp=[zeros((T-t)*NX,NR),K2_temp];
K8_temp=[zeros((T-t)*NX,NR),K8_temp];
A_temp=kron(eye(T-t+1),A);
C_temp=kron(eye(T-t),C);
B_temp=kron(eye(T-t+1),B);
sigma_temp=kron(ones(T-t,1),diag(sigma_eta));
sigma_temp_in=kron(eye(T-t),[sigma_eta(2,2),sigma_eta(1,1)]);
E_temp=kron(eye(T-t+1),E);
K33_temp=kron(diag(K33(i-1,t+1:T)),[0,1;0,0]);
J_value = zeros(1,K);
parfor k=1:K
   X_temp=reshape(squeeze(X_check(:,t:T,k)),(T-t+1)*NX,1);
   X_temp_1=reshape(squeeze(X_check(1,t+1:T,k)),(T-t),1);
   X_temp_2=reshape(squeeze(X_check(2,t+1:T,k)),(T-t),1);
   Eta_temp=reshape(squeeze(Eta_check(:,t+1:T,k)),(T-t)*NX,1);
   R_t=kron(ones(T-t+1,1),squeeze(R_check(:,t,k)));
   [fu,fvalue]=fmincon(@(S) objfun_hd_nl_convex(S,X_temp,R_t,A_temp,B_temp,E_temp,RS_temp,Eta_temp,H3_temp,NX,K1_temp,K2_temp,K8_temp,C_temp,sigma_temp,reshape(u_check(:,:,k),T*NR,1)),R1/T*ones((T-t+1)*NR,1),...
        -eye((T-t+1)*NR),zeros((T-t+1)*NR,1),kron(ones(1,T-t+1),diag(ones(NR,1))),squeeze(R_check(:,t,k))',zeros(NR*(T-t+1),1),kron(ones(T-t+1,1),squeeze(R_check(:,t,k))'),[],options);
    J_value(k)=fvalue-G1_temp'*Eta_temp+H1_exp-Eta_temp'*H1_temp*Eta_temp-2*(C_temp*X_temp(1:end-NX))'*H1_temp*Eta_temp+...
        K3_temp'*(-4*(C_temp*X_temp(1:end-NX)).^3.*Eta_temp+6*(C_temp*X_temp(1:end-NX)).^2.*(sigma_temp-Eta_temp.^2)-4*(C_temp*X_temp(1:end-NX)).*(Eta_temp.^3)+...
        3*sigma_temp.^2-Eta_temp.^4)+K6_temp'*(-3*(C_temp*X_temp(1:end-NX)).^2.*Eta_temp+3*(C_temp*X_temp(1:end-NX)).*(sigma_temp-Eta_temp.^2)-Eta_temp.^3)+...
        ((C_temp*X_temp(1:end-NX)).^2)'*K33_temp*((C_temp*X_temp(1:end-NX)).^2)+K33(i-1,t+1:T)*sigma_temp_in*((C_temp*X_temp(1:end-NX)).^2)+(2*(T-t)*sigma_eta(1,2)^2+(T-t)*sigma_eta(1,1)*sigma_eta(2,2))*sum(K33(i-1,t+1:T))-...
        K33(i-1,t+1:T)*(X_temp_1.^2.*X_temp_2.^2);
end
W_value=mean(J_value)
W_valuevar=var(J_value)

Total_time=etime(clock,t1);
save('max_app.mat')
