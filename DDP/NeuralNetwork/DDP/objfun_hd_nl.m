function F=objfun_hd_nl(S,X_temp,R_t,A_temp,B_temp,E_temp,RS_temp,Eta_temp,H3_temp,NX,K1_temp,K2_temp,K8_temp,C_temp,sigma_temp)
%S=R1/T*ones((T-t+1)*NR,1);

F=S'*E_temp*sqrt(S)+(A_temp*S+B_temp*X_temp)'*(R_t-RS_temp*S)-Eta_temp'*H3_temp*(R_t-RS_temp*S)-...
    Eta_temp'*K1_temp*((R_t-RS_temp*S).^2)+(sigma_temp-2*(C_temp*X_temp(1:end-NX)).*Eta_temp-Eta_temp.^2)'*(K8_temp*(R_t-RS_temp*S)+K2_temp*((R_t-RS_temp*S).^2));