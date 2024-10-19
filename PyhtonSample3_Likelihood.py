import numpy as np
from probs_vector import norm_cdf, norm_inv_cdf
from scipy.stats import norm

def funct(thp, xvar_arr, xtrans_arr, edu_arr, ma1_arr, ma2_arr, tvarm_arr, tvare_arr, n1, u5c, u6c):

    n = 1060
    nt = 6
    sm = 1
    
    agr9 = xvar_arr[:,11]
    compl = np.zeros((n, sm + 1))
    gmx = np.zeros(n)
    cp = 0.0
    wa1 = np.zeros((n, nt))
    wa2 = np.zeros((n, nt))
    wae = np.zeros((n, nt))
    wb1 = np.zeros((n, nt))
    wb2 = np.zeros((n, nt))
    wb3 = np.zeros((n, nt))
    wc1 = np.zeros((n, nt))
    wc2 = np.zeros((n, nt))
    wc3 = np.zeros((n, nt))
    em = np.zeros((n, nt))
    wce = np.zeros((n,nt))
    ma0 = np.zeros((n, nt)) 
    ma0[:, :6] = 1 - ma1_arr[:, :6] - ma2_arr[:, :6] 
    
    xbe = np.zeros((n, nt))
    wbe = np.zeros((n, nt))
    ee = np.zeros((n, nt))
    de = np.zeros((n,nt))
    wcc=np.zeros((n,nt))
   
    gec = thp[34]
    gt = thp[50]
    gam1, gam2 = thp[11 :13]
    kap1, kap2 = thp[14:16]
    game1, game2 = thp[29:31]
    kape1, kape2 = thp[32:34]
    gmg1, gmg2 = thp[63:65]
    sigHS = thp[51]
    rho_m = thp[77]
    rho_a = thp[78]
    
    gmx = xvar_arr[:, :11] @ thp[:11]
    gex = xvar_arr[:, :11] @ thp[18:29]
    gtx = xvar_arr[:, :11] @ thp[35:46]
    ggx = xvar_arr[:, :11] @ thp[52:63]

    complm = np.zeros((n, 7))
    comple = np.zeros((n, 7))
    complm[:, 0] = 1
    comple[:, 0] = 1

    wa_argm = np.zeros((n, 6))
    
    # Redundant re-initialization removed
    cm = np.zeros((n, sm))
    ce = np.zeros((n, sm))
    cgt= np.zeros((n, sm))
    chsg=np.zeros((n, sm))
    
    calltypes = np.zeros(n)
    
    mgi = np.zeros((n,3))
    mgh = np.zeros((n,3))
    
    eta = np.zeros((n,sm,4))
    cdfhsg = np.zeros(n)
    
    for j in range(sm):
        
        eta[:,j,0] = thp[67] * n1[:,j,0]
        eta[:,j,1] = thp[68] * n1[:,j,0] + thp[71] * n1[:,j,1]
        eta[:,j,2] = thp[69] * n1[:,j,0] + thp[72] * n1[:,j,1] + thp[74] * n1[:,j,2]
        eta[:,j,3] = thp[70] * n1[:,j,0] + thp[73] * n1[:,j,1] + thp[75] * n1[:,j,2] + thp[76] * n1[:,j,3]
        
        #**********************************************************************
        # MARIJUANA
        wa_argm[:,0]=gmx + kap1 + kap2 * 0.1 + thp[13] + eta[:,j,0]
        
        wa1[:,0] = norm_cdf(thp[16]-wa_argm[:,0])
        wa2[:,0] = norm_cdf(thp[17]-wa_argm[:,0])
        
        wb1[:,0] = u5c[:,0,j] * wa1[:,0]
        wb2[:,0] = wa1[:,0] + u5c[:,0,j] * (wa2[:,0] - wa1[:,0])
        wb3[:,0] = wa2[:,0] + u5c[:,0,j] * (1 - wa2[:,0])
        
        wb1[:, 0] = np.clip(wb1[:, 0], 0.0001, 0.9999)
        wb2[:, 0] = np.clip(wb2[:, 0], 0.0001, 0.9999)
        wb3[:, 0] = np.clip(wb3[:, 0], 0.0001, 0.9999)
        
        wc1[:,0] = norm_inv_cdf(wb1[:,0])
        wc2[:,0] = norm_inv_cdf(wb2[:,0])
        wc3[:,0] = norm_inv_cdf(wb3[:,0])
        
        em[:,0] = wc1[:,0] * ma0[:,0] + wc2[:,0] * ma1_arr[:,0] + wc3[:,0] * ma2_arr[:,0]
        
        for t in range(1,6):
            wa_argm[:,t] = gam1 * ma1_arr[:,t-1] + gam2 * ma2_arr[:,t-1] + gmx[:] + kap1 * (t+1) + kap2 * (t+1) * (t+1) * 0.1 + thp[13] * edu_arr[:,t-1] + eta[:,j,0] + rho_m * em[:,t-1]
            
            wa1[:,t] = norm_cdf(thp[16]-wa_argm[:,t])
            wa2[:,t] = norm_cdf(thp[17]-wa_argm[:,t])
            
            wb1[:,t] = u5c[:,t,j] * wa1[:,t]
            wb2[:,t] = wa1[:,t] + u5c[:,t,j] * (wa2[:,t] - wa1[:,t])
            wb3[:,t] = wa2[:,t] + u5c[:,t,j] * (1 - wa2[:,t])
            
            wb1[:, t] = np.clip(wb1[:, t], 0.0001, 0.9999)
            wb2[:, t] = np.clip(wb2[:, t], 0.0001, 0.9999)
            wb3[:, t] = np.clip(wb3[:, t], 0.0001, 0.9999)
            
            wc1[:,t] = norm_inv_cdf(wb1[:,t])
            wc2[:,t] = norm_inv_cdf(wb2[:,t])
            wc3[:,t] = norm_inv_cdf(wb3[:,t])
            
            em[:,t] = rho_m * em[:,t-1] + wc1[:,t] * ma0[:,t] + wc2[:,t] * ma1_arr[:,t] + wc3[:,t] * ma2_arr[:,t]
            
        for t in range(6):
            complm[:,t+1] = ((wa1[:,t] * ma0[:,t] + (wa2[:,t] - wa1[:,t]) * ma1_arr[:,t] + (1 - wa2[:,t]) * ma2_arr[:,t]) ** tvarm_arr[:,t]) * complm[:,t]
            
        cm[:,j] = complm[:,6]
        
        #**********************************************************************
        # ATTENDANCE
        de[:,:9] = 2 * edu_arr[:,:9] - 1
        
        xbe[:,0] = gex[:] + gec + thp[31]+ kape1 + kape2 * 0.1 + eta[:,j,1]
        wae[:,0] = norm_cdf(xbe[:,0])
        
        wbe[:,0] = u6c[:,0,j] * wae[:,0]
        wbe[:,0] = np.clip(wbe[:, 0], 0.0001, 0.9999)
        wcc[:,0] = norm_inv_cdf(wbe[:,0])
        wce[:,0] = -de[:,0]*wcc[:,0]
        ee[:,0] = wce[:,0]
        
        for t in range(1,6):
            xbe[:,t] = gex[:] + gec + thp[31] * edu_arr[:,t-1] + kape1 * (t+1) + kape2 * (t+1) * (t+1) * 0.1 + game1 * ma1_arr[:,t-1] + game2 * ma2_arr[:,t-1] + eta[:,j,1] + rho_a * ee[:,t-1]
            wae[:,t] = norm_cdf(xbe[:,t])
            
            wbe[:,t] = u6c[:,t,j] * wae[:,t]
            wbe[:,t] = np.clip(wbe[:, t], 0.0001, 0.9999)
            wcc[:,t] = norm_inv_cdf(wbe[:,t])
            wce[:,t] = -de[:,t]*wcc[:,t]
            ee[:,t] = rho_a * ee[:,t-1] + wce[:,t]

        for t in range(6):
            comple[:, t+1] = (wae[:, t] ** edu_arr[:, t] * (1 - wae[:, t]) ** (1 - edu_arr[:, t]))**tvare_arr[:,t] *comple[:,t]
        
        ce[:,j] = comple[:,6]
        
        #**********************************************************************
        # GPA
        
        for i in range(n):
            
            if agr9[i] == 13:
                mgi[i, 0:3] = ma1_arr[i, 0:3]
                mgh[i, 0:3] = ma2_arr[i, 0:3]
            elif agr9[i] == 14:
                mgi[i, 0:3] = ma1_arr[i, 1:4]
                mgh[i, 0:3] = ma2_arr[i, 1:4]
            elif agr9[i] == 15:
                mgi[i, 0:3] = ma1_arr[i, 2:5]
                mgh[i, 0:3] = ma2_arr[i, 2:5]
            elif agr9[i] == 16:
                mgi[i, 0:3] = ma1_arr[i, 3:6]
                mgh[i, 0:3] = ma2_arr[i, 3:6]
            elif agr9[i] == 17:
                mgi[i, 0:3] = ma1_arr[i, 3:6]
                mgh[i, 0:3] = ma2_arr[i, 3:6]
        
        index10 = gtx + gt + thp[49]*agr9 + thp[46]*mgi[:,0] + thp[47]*mgh[:,0]+ thp[48]*xtrans_arr[:,0] + eta[:,j,2]
        index11 = gtx + gt + thp[49]*agr9 + thp[46]*mgi[:,1] + thp[47]*mgh[:,1]+ thp[48]*xtrans_arr[:,1] + eta[:,j,2]
        index12 = gtx + gt + thp[49]*agr9 + thp[46]*mgi[:,2] + thp[47]*mgh[:,2]+ thp[48]*xtrans_arr[:,2] + eta[:,j,2]
       
        pdf_arg10 = (xtrans_arr[:,1] - index10) / sigHS
        pdf_arg11 = (xtrans_arr[:,2] - index11) / sigHS
        pdf_arg12 = (xtrans_arr[:,3] - index12) / sigHS
        
        wpdf10 = norm.pdf(pdf_arg10)
        wpdf11 = norm.pdf(pdf_arg11)
        wpdf12 = norm.pdf(pdf_arg12)
    
        l10 = (1 / sigHS) * wpdf10
        l11 = (1 / sigHS) * wpdf11
        l12 = (1 / sigHS) * wpdf12

        cgt[:,j] = (l10 ** xtrans_arr[:,7]) * (l11 ** xtrans_arr[:,8]) * (l12 ** xtrans_arr[:,9])
        
        #**********************************************************************
        # High School Graduation
        hsgrad = ggx[:] + thp[66] + thp[65] * agr9 +  gmg1 * xtrans_arr[:,4] + gmg2 * xtrans_arr[:,5] + eta[:,j,3]
        
        cdfhsg[:] = norm_cdf(hsgrad[:])
        
        chsg[:,j] = (cdfhsg[:] ** xtrans_arr[:,6] * (1 - cdfhsg[:]) ** (1 - xtrans_arr[:, 6]))
        
        compl[:,j+1] = cm[:,j]*ce[:,j]*cgt[:,j]*chsg[:,j]+compl[:,j]
        
       # print(chsg[5,j])
       # compl[:,j+1] = cm[:,j]+compl[:,j]
    
    calltypes[:]=compl[:,sm]/sm
        
    for i in range(n):
        if calltypes[i] > 0:
            cp += np.log(calltypes[i])

    fu = -(cp / n)

    return fu
#==============================================================================