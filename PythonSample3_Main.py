
# sample n=1060        
# model M & A and attendance; ordered 
# Born 1983 & 1984, period 1 is age 13
# Using transcript data and added risk index

import numpy as np
import pandas as pd
from timeit import default_timer as timer
from scipy.optimize import minimize
from PyhtonSample3_Likelihood import funct
from tabulate import tabulate
import random


xvar = pd.read_csv('C:/Users/sabar/Downloads/RE_ Fortran code/stata/xvars_p.csv')
edu = pd.read_csv('C:/Users/sabar/Downloads/RE_ Fortran code/stata/att_p.csv')
ma1 = pd.read_csv('C:/Users/sabar/Downloads/RE_ Fortran code/stata/mbin_m.csv')
ma2 = pd.read_csv('C:/Users/sabar/Downloads/RE_ Fortran code/stata/mbin_h.csv')
xtrans = pd.read_csv('C:/Users/sabar/Downloads/RE_ Fortran code/stata/gr_trans_p.csv')
tvare = pd.read_csv('C:/Users/sabar/Downloads/RE_ Fortran code/stata/tvare.csv')
tvarm = pd.read_csv('C:/Users/sabar/Downloads/RE_ Fortran code/stata/tvarm.csv')

xvar_arr = xvar.values
xtrans_arr = xtrans.values
edu_arr = edu.values
ma1_arr = ma1.values
ma2_arr = ma2.values
tvarm_arr = tvarm.values
tvare_arr = tvare.values

n=1060
ns=1
npx=9
np1 = npx + 10  #19
np2 = np1 + 19  #38
np3 = np2 + 17  #55
np4 = np3 + 0   #55
nx = 14
nt = 6
nted = 10
np5 = np4 + 15  #70
npp = np5 + 12  #82
npxs = 22
npxe = 15
sm = 1
nphs = 19

# Set the seed for reproducibility (optional)
random.seed(1234)

u5c = np.zeros((n,nt,sm))
u6c = np.zeros((n,nt,sm))

n1 = np.random.randn(n,sm,4)

print('done with draws')

start = timer()

print('minimize started')
thp = np.zeros(79)
thp[16]=-0.7
thp[17]=0.8
thp[11]=0.8
thp[12]=0.7
thp[13]=-0.7
thp[29]=-0.7
thp[30]=-0.7
thp[31]=0.8
thp[51]=1
thp[67]=0.5
thp[71]=0.5
thp[74]=0.5
thp[76]=0.5
thp[77]=0.3
thp[78]=0.3

res = minimize(funct, thp, args=(xvar_arr, xtrans_arr, edu_arr, ma1_arr, ma2_arr, tvarm_arr, tvare_arr, n1, u5c, u6c), method = 'L-BFGS-B', options={'disp': True, 'maxfun': 100000, 'maxiter':100000})

#THD = np.sqrt(np.diag(res.hess_inv)) / np.sqrt(n)
THD = np.sqrt(np.diag(np.asarray(res.hess_inv.todense()))) / np.sqrt(n)

THT = res.x / THD

print('minimize finished')


f = open("HS_4_UH50_par.txt", "a")

print(res.x, file=f)

print('**********************************************************************', file=f)
print('**********************************************************************', file=f)

f.close()

f1 = open("HS_4_UH50.txt","w")
end = timer()

print (end - start, file=f1)

print('**********************************************************************', file=f1)

table_m = [["Male",res.x[0],THT[0]],
          ["Black",res.x[1],THT[1]],
          ["Hispanic",res.x[2],THT[2]],
          ["Broken",res.x[3],THT[3]],
          ["poor",res.x[4],THT[4]],
          ["Z_math", res.x[5],THT[5]],
          ["Siblings",res.x[6],THT[6]],
          ["Med_dropout",res.x[7],THT[7]],
          ["Med_hs",res.x[8],THT[8]],
          ["Med_coll",res.x[9],THT[9]],
          ["Z_hrisk",res.x[10],THT[10]],
          ["lag_m1",res.x[11],THT[11]],
          ["lag_m2",res.x[12],THT[12]],
          ["lag_e",res.x[13],THT[13]],
          ["age",res.x[14],THT[14]],
          ["age_sq",res.x[15],THT[15]],
          ["const1", res.x[16],THT[16]],
          ["const2", res.x[17],THT[17]]]
 
print(tabulate(table_m, headers=["Marijuana","Estimate", "t-stat"]), file=f1)
print("**********************************************************************", file=f1)
print("**********************************************************************", file=f1)

table_e = [["Male",res.x[18],THT[18]],
          ["Black",res.x[19],THT[19]],
          ["Hispanic",res.x[20],THT[20]],
          ["Broken",res.x[21],THT[21]],
          ["Poor",res.x[22],THT[22]],
          ["Z_math", res.x[23],THT[23]],
          ["Siblings",res.x[24],THT[24]],
          ["Med_dropout",res.x[25],THT[25]],
          ["Med_hs",res.x[26],THT[26]],
          ["Med_coll",res.x[27],THT[27]],
          ["Z_hrisk",res.x[28],THT[28]],
          ["lag_m1",res.x[29],THT[29]],
          ["lag_m2",res.x[30],THT[30]],
          ["lag_e",res.x[31],THT[31]],
          ["age",res.x[32],THT[32]],
          ["age_sq",res.x[33],THT[33]],
          ["consta2", res.x[34],THT[34]]]
 
print(tabulate(table_e, headers=["Attendance","Estimate", "t-stat"]), file=f1)
print("**********************************************************************", file=f1)
print("**********************************************************************", file=f1)

table_tr = [["Male",res.x[35],THT[35]],
          ["Black",res.x[36],THT[36]],
          ["Hispanic",res.x[37],THT[37]],
          ["Broken",res.x[38],THT[38]],
          ["Poor",res.x[39],THT[39]],
          ["Z_math", res.x[40],THT[40]],
          ["Siblings",res.x[41],THT[41]],
          ["Med_dropout",res.x[42],THT[42]],
          ["Med_hs",res.x[43],THT[43]],
          ["Med_coll",res.x[44],THT[44]],
          ["Z_hrisk",res.x[45],THT[45]],
          ["lag_m1",res.x[46],THT[46]],
          ["lag_m2",res.x[47],THT[47]],
          ["lag_GPA",res.x[48],THT[48]],
          ["age in grade 9",res.x[49],THT[49]],
          ["const", res.x[50],THT[50]],
          ["sigmaHS",res.x[51],THT[51]]]
          
print(tabulate(table_tr, headers=["GPA","Estimate", "t-stat"]), file=f1)
print("**********************************************************************", file=f1)
print("**********************************************************************", file=f1)

table_grad = [["Male",res.x[52],THT[52]],
          ["Black",res.x[53],THT[53]],
          ["Hispanic",res.x[54],THT[54]],
          ["Broken",res.x[55],THT[55]],
          ["Poor",res.x[56],THT[56]],
          ["Z_math", res.x[57],THT[57]],
          ["Siblings",res.x[58],THT[58]],
          ["Med_dropout",res.x[59],THT[59]],
          ["Med_hs",res.x[60],THT[60]],
          ["Med_coll",res.x[61],THT[61]],
          ["Z_hrisk",res.x[62],THT[62]],
          ["smHS_med",res.x[63],THT[63]],
          ["smHS_high",res.x[64],THT[64]],
          ["age in grade 9",res.x[65],THT[65]],
          ["const",res.x[66],THT[66]]]
          
print(tabulate(table_grad, headers=["HS graduation","Estimate", "t-stat"]), file=f1)
print("**********************************************************************", file=f1)
print("**********************************************************************", file=f1)

table_sig = [["Sigma_m",res.x[67],THT[67]],
          ["Sigma_me",res.x[68],THT[68]],
          ["Sigma_mt",res.x[69],THT[69]],
          ["Sigma_mhs",res.x[70],THT[70]],
          ["Sigma_e",res.x[71],THT[71]],
          ["Sigma_et", res.x[72],THT[72]],
          ["Sigma_ehs",res.x[73],THT[73]],
          ["Sigma_t",res.x[74],THT[74]],
          ["Sigma_ths",res.x[75],THT[75]],
          ["Sigma_hs",res.x[76],THT[76]],
          ["rho_m",res.x[77],THT[77]],
          ["rho_a",res.x[78],THT[78]]]
          
print(tabulate(table_sig, headers=["Variance Params","Estimate", "t-stat"]), file=f1)

chol1 = np.zeros((4,4))
std_dev = np.zeros((4, 1))
correl = np.zeros((4,4))

chol1[0,0] = res.x[67]
chol1[1,0] = res.x[68]
chol1[2,0] = res.x[69]
chol1[3,0] = res.x[70]

chol1[1,1] = res.x[71]
chol1[2,1] = res.x[72]
chol1[3,1] = res.x[73]

chol1[2,2] = res.x[74]
chol1[3,2] = res.x[75]

chol1[3,3] = res.x[76]

chol1t = chol1.transpose()
covvar = np.matmul(chol1, chol1t)

print('**********************************************************************', file=f1)
print(covvar, file=f1)


# for i in range(1):
std_dev[0] = np.sqrt(covvar[0, 0])
std_dev[1] = np.sqrt(covvar[1, 1])
std_dev[2] = np.sqrt(covvar[2, 2])
std_dev[3] = np.sqrt(covvar[3, 3])

print('**********************************************************************', file=f1)
print(std_dev, file=f1)

#for i in range(1):
#    for j in range(1):
correl[0, 0] = covvar[0, 0] / (std_dev[0] * std_dev[0])
correl[1, 0] = covvar[1, 0] / (std_dev[1] * std_dev[0])
correl[2, 0] = covvar[2, 0] / (std_dev[2] * std_dev[0])
correl[3, 0] = covvar[3, 0] / (std_dev[3] * std_dev[0])

correl[1, 1] = covvar[1, 1] / (std_dev[1] * std_dev[1])
correl[2, 1] = covvar[2, 1] / (std_dev[2] * std_dev[1])
correl[3, 1] = covvar[3, 1] / (std_dev[3] * std_dev[1])

correl[2, 2] = covvar[2, 2] / (std_dev[2] * std_dev[2])
correl[3, 2] = covvar[3, 2] / (std_dev[3] * std_dev[2])

correl[3, 3] = covvar[3, 3] / (std_dev[3] * std_dev[3])


print('**********************************************************************', file=f1)
print(correl, file=f1)

f1.close()