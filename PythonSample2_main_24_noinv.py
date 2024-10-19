import numpy as np
import pandas as pd
from scipy.optimize import minimize
from PythonSample2_likelihood_24_noinv import funct
from numpy.linalg import LinAlgError
from tabulate import tabulate

# Define global constants and arrays
NS = 2615  # number of observations
NP = 72    # number of parameters to be estimated

# Initialize arrays
MC = np.zeros((NS, 9))
MNC = np.zeros((NS, 6))
MA = np.zeros((NS, 13))
MNA = np.zeros((NS, 39))
ALPHA1 = np.zeros((26, 4))
ALPHA2 = np.zeros((22, 4))
ALPHA3 = np.zeros((19, 4))
MEDU = np.zeros((NS, 5))
SIGMA = np.zeros((12, 12))
F_1 = np.zeros((26, 26))
F_2 = np.zeros((22, 22))
F_3 = np.zeros((19, 19))
Q1 = np.zeros((4, 4))
Q2 = np.zeros((4, 4))
K1 = np.zeros((26, 26))
K2 = np.zeros((22, 22))
K3 = np.zeros((19, 19))
F1INV = np.zeros((26, 26))
A1 = np.zeros(4)
P1 = np.zeros((4, 4))
Y1 = np.zeros((NS, 26))
Y2 = np.zeros((NS, 22))
Y3 = np.zeros((NS, 19))
COV_ALL = np.zeros((67, 67))

MC = pd.read_csv('C:/Users/sabar/source/repos/skill_extracurricular/mc.csv')
MNC = pd.read_csv('C:/Users/sabar/source/repos/skill_extracurricular/mnc.csv')
MA = pd.read_csv('C:/Users/sabar/source/repos/skill_extracurricular/ma.csv')
MNA = pd.read_csv('C:/Users/sabar/source/repos/skill_extracurricular/mna.csv')
MI = pd.read_csv('C:/Users/sabar/source/repos/skill_extracurricular/mi.csv')
MEDU = pd.read_csv('C:/Users/sabar/source/repos/skill_extracurricular/car.csv')
CC = pd.read_csv('C:/Users/sabar/source/repos/skill_extracurricular/cov.csv')

mc = MC.values
mnc = MNC.values
mi = MI.values
ma = MA.values
mna = MNA.values
medu = MEDU.values
COV_ALL = CC.values


# Vectorize Y1 and Y2 assignment
Y1[:, 0:3] = mc[:, 0:3]
Y1[:, 3:5] = mnc[:, 0:2]
Y1[:, 5:13] = ma[:, 0:8]
Y1[:, 13:26]= mna[:, 0:13]

Y2[:, 0:3] = mc[:, 3:6]
Y2[:, 3:5] = mnc[:, 2:4]
Y2[:, 5:7] = ma[:, 8:10]
Y2[:, 7:22]= mna[:, 13:28]

Y3[:, 0:3] = mc[:, 6:9]
Y3[:, 3:5] = mnc[:, 4:6]
Y3[:, 5:8] = ma[:, 10:13]
Y3[:, 8:19]= mna[:, 28:39]

# ALPHA1 assignment
ALPHA1[0, 0] = 1
ALPHA1[3, 1] = 1
ALPHA1[5, 2] = 1
ALPHA1[13,3] = 1
ALPHA1[1, 0] = COV_ALL[1, 3] / COV_ALL[0, 3]
ALPHA1[2, 0] = COV_ALL[2, 3] / COV_ALL[0, 3]
ALPHA1[4, 1] = COV_ALL[10, 11] / COV_ALL[9, 11]
ALPHA1[6, 2] = COV_ALL[16, 23] / COV_ALL[23, 15]
ALPHA1[7, 2] = COV_ALL[17, 23] / COV_ALL[23, 15]
ALPHA1[8, 2] = COV_ALL[18, 23] / COV_ALL[23, 15]
ALPHA1[9, 2] = COV_ALL[19, 23] / COV_ALL[23, 15]
ALPHA1[10, 2] = COV_ALL[20, 23] / COV_ALL[23, 15]
ALPHA1[11, 2] = COV_ALL[21, 23] / COV_ALL[23, 15]
ALPHA1[12, 2] = COV_ALL[22, 23] / COV_ALL[23, 15]
ALPHA1[14, 3] = COV_ALL[29, 41] / COV_ALL[41, 28]
ALPHA1[15, 3] = COV_ALL[30, 41] / COV_ALL[41, 28]
ALPHA1[16, 3] = COV_ALL[31, 41] / COV_ALL[41, 28]
ALPHA1[17, 3] = COV_ALL[32, 41] / COV_ALL[41, 28]
ALPHA1[18, 3] = COV_ALL[33, 41] / COV_ALL[41, 28]
ALPHA1[19, 3] = COV_ALL[34, 41] / COV_ALL[41, 28]
ALPHA1[20, 3] = COV_ALL[35, 41] / COV_ALL[41, 28]
ALPHA1[21, 3] = COV_ALL[36, 41] / COV_ALL[41, 28]
ALPHA1[22, 3] = COV_ALL[37, 41] / COV_ALL[41, 28]
ALPHA1[23, 3] = COV_ALL[38, 41] / COV_ALL[41, 28]
ALPHA1[24, 3] = COV_ALL[39, 41] / COV_ALL[41, 28]
ALPHA1[25, 3] = COV_ALL[40, 41] / COV_ALL[41, 28]



# ALPHA2 assignment
ALPHA2[0, 0] = 1
ALPHA2[3, 1] = 1
ALPHA2[5, 2] = 1
ALPHA2[7, 3] = 1
ALPHA2[1, 0] = COV_ALL[4, 6] / COV_ALL[6, 3]
ALPHA2[2, 0] = COV_ALL[5, 6] / COV_ALL[6, 3]
ALPHA2[4, 1] = COV_ALL[12, 13] / COV_ALL[13, 11]
ALPHA2[6, 2] = COV_ALL[24, 25] / COV_ALL[25, 23]
ALPHA2[8, 3] = COV_ALL[42, 56] / COV_ALL[56, 41]
ALPHA2[9, 3] = COV_ALL[43, 56] / COV_ALL[56, 41]
ALPHA2[10, 3] = COV_ALL[44, 56] / COV_ALL[56, 41]
ALPHA2[11, 3] = COV_ALL[45, 56] / COV_ALL[56, 41]
ALPHA2[12, 3] = COV_ALL[46, 56] / COV_ALL[56, 41]
ALPHA2[13, 3] = COV_ALL[47, 56] / COV_ALL[56, 41]
ALPHA2[14, 3] = COV_ALL[48, 56] / COV_ALL[56, 41]
ALPHA2[15, 3] = COV_ALL[49, 56] / COV_ALL[56, 41]
ALPHA2[16, 3] = COV_ALL[50, 56] / COV_ALL[56, 41]
ALPHA2[17, 3] = COV_ALL[51, 56] / COV_ALL[56, 41]
ALPHA2[18, 3] = COV_ALL[52, 56] / COV_ALL[56, 41]
ALPHA2[19, 3] = COV_ALL[53, 56] / COV_ALL[56, 41]
ALPHA2[20, 3] = COV_ALL[54, 56] / COV_ALL[56, 41]
ALPHA2[21, 3] = COV_ALL[55, 56] / COV_ALL[56, 41]

ALPHA3[0, 0] = 1
ALPHA3[3, 1] = 1
ALPHA3[5, 2] = 1
ALPHA3[8,3] = 1
ALPHA3[1, 0] = COV_ALL[7, 3] / COV_ALL[6, 3]
ALPHA3[2, 0] = COV_ALL[8, 3] / COV_ALL[6, 3]
ALPHA3[4, 1] = COV_ALL[14, 11] / COV_ALL[13, 11]
ALPHA3[6, 2] = COV_ALL[26, 23] / COV_ALL[25, 23]
ALPHA3[7, 2] = COV_ALL[27, 23] / COV_ALL[25, 23]
ALPHA3[9, 3] = COV_ALL[57, 41] / COV_ALL[56, 41]
ALPHA3[10, 3] = COV_ALL[58, 41] / COV_ALL[56, 41]
ALPHA3[11, 3] = COV_ALL[59, 41] / COV_ALL[56, 41]
ALPHA3[12, 3] = COV_ALL[60, 41] / COV_ALL[56, 41]
ALPHA3[13, 3] = COV_ALL[61, 41] / COV_ALL[56, 41]
ALPHA3[14, 3] = COV_ALL[62, 41] / COV_ALL[56, 41]
ALPHA3[15, 3] = COV_ALL[63, 41] / COV_ALL[56, 41]
ALPHA3[16, 3] = COV_ALL[64, 41] / COV_ALL[56, 41]
ALPHA3[17, 3] = COV_ALL[65, 41] / COV_ALL[56, 41]
ALPHA3[18, 3] = COV_ALL[66, 41] / COV_ALL[56, 41]


A1 = np.array([1.38e-07,-1.05e-09,1.16e-08,4.01e-08])

# SIGMA assignment
SIGMA[0, :] = [COV_ALL[0, 1] / ALPHA1[1, 0], COV_ALL[0, 9], COV_ALL[0, 15], COV_ALL[0, 28], COV_ALL[0, 3], COV_ALL[0, 11], COV_ALL[0, 23], COV_ALL[0, 41], COV_ALL[0, 6], COV_ALL[0, 13], COV_ALL[0, 25], COV_ALL[0, 56]]
SIGMA[1, :] = [SIGMA[0, 1], COV_ALL[9, 10] / ALPHA1[4, 1], COV_ALL[9, 15], COV_ALL[9,28], COV_ALL[9,3], COV_ALL[9,11], COV_ALL[9,23], COV_ALL[9,41], COV_ALL[9,6], COV_ALL[9,13], COV_ALL[9,25], COV_ALL[9,56]]
SIGMA[2, :] = [SIGMA[0, 2], SIGMA[1, 2], COV_ALL[15, 16] / ALPHA1[6, 2], COV_ALL[15, 28], COV_ALL[15, 3], COV_ALL[15,11], COV_ALL[15,23], COV_ALL[15,41], COV_ALL[15,6], COV_ALL[15, 13], COV_ALL[13, 25], COV_ALL[15,56]]
SIGMA[3, :] = [SIGMA[0, 3], SIGMA[1, 3], SIGMA[2, 3], COV_ALL[28, 29]/ ALPHA1[14, 3], COV_ALL[28, 3], COV_ALL[28, 11], COV_ALL[28, 23], COV_ALL[28, 41], COV_ALL[28,6], COV_ALL[28, 13], COV_ALL[28, 25], COV_ALL[28,56]]
SIGMA[4, :] = [SIGMA[0, 4], SIGMA[1, 4], SIGMA[2, 4], SIGMA[3, 4], COV_ALL[3, 4]/ALPHA2[1, 0], COV_ALL[3, 11], COV_ALL[3, 23], COV_ALL[3, 41], COV_ALL[3, 6], COV_ALL[3,13], COV_ALL[3, 25], COV_ALL[3, 56]]
SIGMA[5, :] = [SIGMA[0, 5], SIGMA[1, 5], SIGMA[2, 5], SIGMA[3, 5], SIGMA[4, 5], COV_ALL[11, 12]/ALPHA2[4, 1], COV_ALL[11, 23], COV_ALL[11,41], COV_ALL[11,6], COV_ALL[11,13], COV_ALL[11,25], COV_ALL[11,56]]
SIGMA[6, :] = [SIGMA[0, 6], SIGMA[1, 6], SIGMA[2, 6], SIGMA[3, 6], SIGMA[4, 6], SIGMA[5, 6], COV_ALL[23, 24]/ALPHA2[6, 2], COV_ALL[23, 41], COV_ALL[23,6], COV_ALL[23,13], COV_ALL[23,25], COV_ALL[23,56]]
SIGMA[7, :] = [SIGMA[0, 7], SIGMA[1, 7], SIGMA[2, 7], SIGMA[3, 7], SIGMA[4, 7], SIGMA[5, 7], SIGMA[6, 7], COV_ALL[41, 42]/ALPHA2[8, 3], COV_ALL[41, 6], COV_ALL[41, 13], COV_ALL[41,25], COV_ALL[41, 56]]
SIGMA[8, :] = [SIGMA[0, 8], SIGMA[1, 8], SIGMA[2, 8], SIGMA[3, 8], SIGMA[4, 8], SIGMA[5, 8], SIGMA[6, 8], SIGMA[7, 8], COV_ALL[6, 7]/ALPHA3[1,0], COV_ALL[6, 13], COV_ALL[6,25], COV_ALL[6, 56]]
SIGMA[9, :] = [SIGMA[0, 9], SIGMA[1, 9], SIGMA[2, 9], SIGMA[3, 9], SIGMA[4, 9], SIGMA[5, 9], SIGMA[6, 9], SIGMA[7, 9], SIGMA[8, 9], COV_ALL[13, 14]/ALPHA3[4, 1], COV_ALL[13,25], COV_ALL[13, 56]]
SIGMA[10,:] = [SIGMA[0,10], SIGMA[1,10], SIGMA[2,10], SIGMA[3,10], SIGMA[4,10], SIGMA[5,10], SIGMA[6,10], SIGMA[7,10], SIGMA[8,10], SIGMA[9,10], COV_ALL[25, 26]/ALPHA3[6, 2], COV_ALL[25, 56]]
SIGMA[11,:] = [SIGMA[0,11], SIGMA[1,11], SIGMA[2,11], SIGMA[3,11], SIGMA[4,11], SIGMA[5,11], SIGMA[6,11], SIGMA[7,11], SIGMA[8,11], SIGMA[9,11], SIGMA[10,11], COV_ALL[56,57]/ALPHA3[9, 3]]


# K1 assignment
K1[0, 0] = COV_ALL[0, 0] - SIGMA[0, 0]
K1[1, 1] = COV_ALL[1, 1] - (ALPHA1[1, 0]**2) * SIGMA[0, 0]
K1[2, 2] = COV_ALL[2, 2] - (ALPHA1[2, 0]**2) * SIGMA[0, 0]
K1[3, 3] = COV_ALL[9, 9] - SIGMA[1, 1]
K1[4, 4] = COV_ALL[10, 10] - (ALPHA1[4, 1]**2) * SIGMA[1, 1]
K1[5, 5] = COV_ALL[15, 15] - SIGMA[2, 2]
K1[6, 6] = COV_ALL[16, 16] - (ALPHA1[6, 2]**2) * SIGMA[2, 2]
K1[7, 7] = COV_ALL[17, 17] - (ALPHA1[7, 2]**2) * SIGMA[2, 2]
K1[8, 8] = COV_ALL[18, 18] - (ALPHA1[8, 2]**2) * SIGMA[2, 2]
K1[9, 9] = COV_ALL[19, 19] - (ALPHA1[9, 2]**2) * SIGMA[2, 2]
K1[10, 10] = COV_ALL[20, 20] - (ALPHA1[10, 2]**2) * SIGMA[2, 2]
K1[11, 11] = COV_ALL[21, 21] - (ALPHA1[11, 2]**2) * SIGMA[2, 2]
K1[12, 12] = COV_ALL[22, 22] - (ALPHA1[12, 2]**2) * SIGMA[2, 2]
K1[13, 13] = COV_ALL[28, 28] - SIGMA[3, 3]
K1[14, 14] = COV_ALL[29, 29] - (ALPHA1[14, 3]**2) * SIGMA[3, 3]
K1[15, 15] = COV_ALL[30, 30] - (ALPHA1[15, 3]**2) * SIGMA[3, 3]
K1[16, 16] = COV_ALL[31, 31] - (ALPHA1[16, 3]**2) * SIGMA[3, 3]
K1[17, 17] = COV_ALL[32, 32] - (ALPHA1[17, 3]**2) * SIGMA[3, 3]
K1[18, 18] = COV_ALL[33, 33] - (ALPHA1[18, 3]**2) * SIGMA[3, 3]
K1[19, 19] = COV_ALL[34, 34] - (ALPHA1[19, 3]**2) * SIGMA[3, 3]
K1[20, 20] = COV_ALL[35, 35] - (ALPHA1[20, 3]**2) * SIGMA[3, 3]
K1[21, 21] = COV_ALL[36, 36] - (ALPHA1[21, 3]**2) * SIGMA[3, 3]
K1[22, 22] = COV_ALL[37, 37] - (ALPHA1[22, 3]**2) * SIGMA[3, 3]
K1[23, 23] = COV_ALL[38, 38] - (ALPHA1[23, 3]**2) * SIGMA[3, 3]
K1[24, 24] = COV_ALL[39, 39] - (ALPHA1[24, 3]**2) * SIGMA[3, 3]
K1[25, 25] = COV_ALL[40, 40] - (ALPHA1[25, 3]**2) * SIGMA[3, 3]


# Initial parameters for minimization
X = np.zeros(72)
X[0]=1

# Minimize the function
try:
    res = minimize(funct, X, args=(Y1, Y2, Y3, A1, SIGMA, ALPHA1, ALPHA2, ALPHA3, COV_ALL, K1, medu), method='L-BFGS-B', options={'disp': True, 'maxfun': 100000, 'maxiter': 100000, 'ftol':1e-6})
    print("Optimization Result:", res)
except LinAlgError as e:
    print("Linear Algebra Error during optimization:", e)
except ValueError as e:
    print("Value Error during optimization:", e)
except Exception as e:
    print("Unexpected error during optimization:", e)

THD = np.sqrt(np.diag(np.asarray(res.hess_inv.todense()))) / np.sqrt(NS)

THT = res.x / THD

print('minimize finished')

f = open("RESULTS_PYTHON.txt", "a")

print(res.x, file=f)

print('**********************************************************************', file=f)
print('**********************************************************************', file=f)

f.close()

f1 = open("table1.txt","w")

table_m = [["Cog14Cog16",res.x[0],THT[0]],
          ["nonCog14Cog16",res.x[1],THT[1]],
          ["acad14Cog16",res.x[2],THT[2]],
          ["nonacad14Cog16",res.x[3],THT[3]],
          ["meduCog16",res.x[4],THT[4]],
          ["hispanicCog16",res.x[5],THT[5]],
          ["blackCog16",res.x[6],THT[6]],
          ["ruralCog16",res.x[7],THT[7]],
          ["maleCog16",res.x[8],THT[8]],
          ["Cog14nonCog16",res.x[9],THT[9]],
          ["nonCog14nonCog16", res.x[10],THT[10]],
          ["acad14nonCog16",res.x[11],THT[11]],
          ["nonacad14nonCog16",res.x[12],THT[12]],
          ["medunonCog16",res.x[13],THT[13]],
          ["hispanicnonCog16",res.x[14],THT[14]],
          ["blacknonCog16",res.x[15],THT[15]],
          ["ruralCog16",res.x[16],THT[16]],
          ["malenonCog16",res.x[17],THT[17]],
          ["Cog14acad16",res.x[18],THT[18]],
          ["nonCog14acad16",res.x[19],THT[19]],
          ["acad14acad16",res.x[20],THT[20]],
          ["nonacad14acad16",res.x[21],THT[21]],
          ["meduacad16",res.x[22],THT[22]],
          ["hispanicacad16",res.x[23],THT[23]],
          ["blackacad16",res.x[24],THT[24]],
          ["ruralacad16",res.x[25],THT[25]],
          ["maleacad16",res.x[26],THT[26]],
          ["Cog14nonacad16",res.x[27],THT[27]],
          ["nonCog14nonacad16",res.x[28],THT[28]],
          ["acad14nonacad16",res.x[29],THT[29]],
          ["nonacad14nonacad16",res.x[30],THT[30]],
          ["medunonacad16",res.x[31],THT[31]],
          ["hispanicnonacad16",res.x[32],THT[32]],
          ["blacknonacad16",res.x[33],THT[33]],
          ["ruralnonacad16",res.x[34],THT[34]],
          ["malenonacad16",res.x[35],THT[35]],
          ["Cog16Cog18",res.x[36],THT[36]],
          ["nonCog16Cog18",res.x[37],THT[37]],
          ["acad16Cog18",res.x[38],THT[38]],
          ["nonacad16Cog18",res.x[39],THT[39]],
          ["meduCog18",res.x[40],THT[40]],
          ["hispanicCog18",res.x[41],THT[41]],
          ["blackCog18",res.x[42],THT[42]],
          ["ruralCog18",res.x[43],THT[43]],
          ["maleCog18",res.x[44],THT[44]],
          ["Cog16nonCog18",res.x[45],THT[45]],
          ["nonCog16nonCog18", res.x[46],THT[46]],
          ["acad16nonCog18",res.x[47],THT[47]],
          ["nonacad16nonCog18",res.x[48],THT[48]],
          ["medunonCog18",res.x[49],THT[49]],
          ["hispanicnonCog18",res.x[50],THT[50]],
          ["blacknonCog18",res.x[51],THT[51]],
          ["ruralnonCog18",res.x[52],THT[52]],
          ["malenonCog18",res.x[53],THT[53]],
          ["Cog16acad18",res.x[54],THT[54]],
          ["nonCog16acad18",res.x[55],THT[55]],
          ["acad16acad18",res.x[56],THT[56]],
          ["nonacad16acad18",res.x[57],THT[57]],
          ["meduacad18",res.x[58],THT[58]],
          ["hispanicacad18",res.x[59],THT[59]],
          ["blackacad18",res.x[60],THT[60]],
          ["ruralacad18",res.x[61],THT[61]],
          ["maleacad18",res.x[62],THT[62]],
          ["Cog16nonacad18",res.x[63],THT[63]],
          ["nonCog16nonacad18",res.x[64],THT[64]],
          ["acad16nonacad18",res.x[65],THT[65]],
          ["nonacad16nonacad18",res.x[66],THT[66]],
          ["medunonacad18",res.x[67],THT[67]],
          ["hispanicnonacad18",res.x[68],THT[68]],
          ["blacknonacad18",res.x[69],THT[69]],
          ["ruralnonacad18",res.x[70],THT[70]],
          ["malenonacad18",res.x[71],THT[71]]]
 
print(tabulate(table_m, headers=["Coeffs","Estimate", "t-stat"]), file=f1)
print('**********************************************************************', file=f1)

f1.close()
