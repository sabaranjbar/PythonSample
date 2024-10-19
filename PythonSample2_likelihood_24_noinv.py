import numpy as np
from numba import njit

@njit
def make_positive_definite(matrix):
    """Ensure the matrix is positive definite by adding a small value to the diagonal."""
    min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
    if min_eig < 0:
        matrix -= 10 * min_eig * np.eye(*matrix.shape)
    return matrix

@njit
def allclose(a, b, rtol=1e-5, atol=1e-8):
    return np.all(np.abs(a - b) <= (atol + rtol * np.abs(b)))

@njit
def multivariate_normal_pdf(y, mean, cov):
    n = len(mean)
    diff = y - mean
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** n * det_cov))
    exp_term = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
    return norm_const * np.exp(exp_term)

@njit(cache=True, fastmath=True)
def funct(X, Y1, Y2, Y3, A1, SIGMA, ALPHA1, ALPHA2, ALPHA3, COV_ALL, K1, MEDU):
    NS = 2615
    G1 = np.zeros((4, 4))
    G2 = np.zeros((4, 4))
    B1 = np.zeros((4, 5))
    B2 = np.zeros((4, 5))
    
    G1[0, :4] = X[:4]
    B1[0, :5] = X[4:9]
    G1[1, :4] = X[9:13]
    B1[1, :5] = X[13:18]
    G1[2, :4] = X[18:22]
    B1[2, :5] = X[22:27]
    G1[3, :4] = X[27:31]
    B1[3, :5] = X[31:36]
    G2[0, :4] = X[36:40]
    B2[0, :5] = X[40:45]
    G2[1, :4] = X[45:49]
    B2[1, :5] = X[49:54]
    G2[2, :4] = X[54:58]
    B2[2, :5] = X[58:63]
    G2[3, :4] = X[63:67]
    B2[3, :5] = X[67:72]
    
    VAR_EPSILON_COG_1 = COV_ALL[0, 0] - SIGMA[0, 0]
    VAR_EPSILON_NON_1 = COV_ALL[9,9] - SIGMA[1, 1]
    VAR_EPSILON_ACA_1 = COV_ALL[15,15] - SIGMA[2, 2]
    VAR_EPSILON_NAC_1 = COV_ALL[28,28] - SIGMA[3, 3]
    
    VAR_EPSILON_COG_2 = COV_ALL[3, 3] - SIGMA[4, 4]
    VAR_EPSILON_NON_2 = COV_ALL[11,11] - SIGMA[5, 5]
    VAR_EPSILON_ACA_2 = COV_ALL[23,23] - SIGMA[6, 6]
    VAR_EPSILON_NAC_2 = COV_ALL[41,41] - SIGMA[7, 7]
    
    VAR_EPSILON_COG_3 = COV_ALL[6, 6] - SIGMA[8, 8]
    VAR_EPSILON_NON_3 = COV_ALL[13,13] - SIGMA[9, 9]
    VAR_EPSILON_ACA_3 = COV_ALL[25,25] - SIGMA[10, 10]
    VAR_EPSILON_NAC_3 = COV_ALL[56, 56] - SIGMA[11, 11]

    Q1 = np.zeros((4, 4))
    Q1[0, 0] = 0.6510154**2 - VAR_EPSILON_COG_2 - (0.9594814**2) * VAR_EPSILON_COG_1 - (0.0446725**2) * VAR_EPSILON_NON_1
    Q1[1, 1] = 0.9318603**2 - VAR_EPSILON_NON_2 - (0.6498436**2) * VAR_EPSILON_NON_1
    Q1[2, 2] = 1.0858860**2 - VAR_EPSILON_ACA_2 - (0.2346324**2) * VAR_EPSILON_COG_1 - (0.1340775**2) * VAR_EPSILON_NON_1 - (0.4815866**2) * VAR_EPSILON_ACA_1 - (0.1578459**2) * VAR_EPSILON_NAC_1
    Q1[3, 3] = 1.0012290**2 - VAR_EPSILON_NAC_2 - (0.0828729**2) * VAR_EPSILON_COG_1 - (0.3209362**2) * VAR_EPSILON_NAC_1
    
    Q2 = np.zeros((4, 4))
    Q2[0, 0] = 0.6430611**2 - VAR_EPSILON_COG_3 - (0.9184039**2) * VAR_EPSILON_COG_2 - (0.0708840**2) * VAR_EPSILON_NAC_2
    Q2[1, 1] = 0.8951637**2 - VAR_EPSILON_NON_3 - (0.0783412**2) * VAR_EPSILON_COG_2 - (0.6452513**2) * VAR_EPSILON_NON_2
    Q2[3, 3] = 1.0654100**2 - VAR_EPSILON_ACA_3 - (0.2383394**2) * VAR_EPSILON_COG_2 - (0.8954920**2) * VAR_EPSILON_ACA_2 - (0.1107025**2) * VAR_EPSILON_NAC_2
    Q2[4, 4] = 1.3924950**2 - VAR_EPSILON_NAC_3 - (0.1226890**2) * VAR_EPSILON_COG_2 - (0.1022038**2) * VAR_EPSILON_NON_2 - (1.2543520**2) * VAR_EPSILON_NAC_2
    
    P1 = SIGMA[:4, :4]
    F_1 = np.dot(np.dot(ALPHA1, P1), ALPHA1.T) + K1

    F_1 = (F_1 + F_1.T) / 2
    F_1 = make_positive_definite(F_1)

    if not allclose(F_1, F_1.T):
        raise ValueError("Covariance matrix F_1 is not symmetric")

    if not np.all(np.linalg.eigvals(F_1) > 0):
        raise ValueError("Covariance matrix F_1 is not positive definite")

    F1INV = np.linalg.inv(F_1)
    
    L = np.zeros(NS)
    for I in range(NS):
        mean1 = np.dot(ALPHA1, A1)
        cov1 = F_1
        F1 = multivariate_normal_pdf(Y1[I, :26], mean=mean1, cov=cov1)

        A2 = (np.dot(G1, A1) + np.dot(G1, np.dot(P1, np.dot(ALPHA1.T, np.dot(F1INV, (Y1[I, :26] - np.dot(ALPHA1, A1)))))) + B1 @ MEDU[I,:])

        P2 = (np.dot(G1, np.dot(P1, G1.T)) - np.dot(G1, np.dot(P1, np.dot(ALPHA1.T, np.dot(F1INV, np.dot(ALPHA1, np.dot(P1, G1.T)))))) + Q1)

        K2 = np.zeros((22, 22))
        K2[0, 0] = COV_ALL[3, 3] - P2[0, 0]
        K2[1, 1] = COV_ALL[4, 4] - (ALPHA2[1, 0] ** 2) * P2[0, 0]
        K2[2, 2] = COV_ALL[5, 5] - (ALPHA2[2, 0] ** 2) * P2[0, 0]
        K2[3, 3] = COV_ALL[11, 11] - P2[1, 1]
        K2[4, 4] = COV_ALL[12, 12] - (ALPHA2[4, 1] ** 2) * P2[1, 1]
        K2[5, 5] = COV_ALL[23, 23] - P2[2, 2]
        K2[6, 6] = COV_ALL[24, 24] - (ALPHA2[6, 2] ** 2) * P2[2, 2]
        K2[7, 7] = COV_ALL[41, 41] - P2[3, 3]
        K2[8, 8] = COV_ALL[42, 42] - (ALPHA2[8, 3] ** 2) * P2[3, 3]
        K2[9, 9] = COV_ALL[43, 43] - (ALPHA2[9, 3] ** 2) * P2[3, 3]
        K2[10, 10] = COV_ALL[44, 44] - (ALPHA2[10, 3] ** 2) * P2[3, 3]
        K2[11, 11] = COV_ALL[45, 45] - (ALPHA2[11, 3] ** 2) * P2[3, 3]
        K2[12, 12] = COV_ALL[46, 46] - (ALPHA2[12, 3] ** 2) * P2[3, 3]
        K2[13, 13] = COV_ALL[47, 47] - (ALPHA2[13, 3] ** 2) * P2[3, 3]
        K2[14, 14] = COV_ALL[48, 48] - (ALPHA2[14, 3] ** 2) * P2[3, 3]
        K2[15, 15] = COV_ALL[49, 49] - (ALPHA2[15, 3] ** 2) * P2[3, 3]
        K2[16, 16] = COV_ALL[50, 50] - (ALPHA2[16, 3] ** 2) * P2[3, 3]
        K2[17, 17] = COV_ALL[51, 51] - (ALPHA2[17, 3] ** 2) * P2[3, 3]
        K2[18, 18] = COV_ALL[52, 52] - (ALPHA2[18, 3] ** 2) * P2[3, 3]
        K2[19, 19] = COV_ALL[53, 53] - (ALPHA2[19, 3] ** 2) * P2[3, 3]
        K2[20, 20] = COV_ALL[54, 54] - (ALPHA2[20, 3] ** 2) * P2[3, 3]
        K2[21, 21] = COV_ALL[55, 55] - (ALPHA2[21, 3] ** 2) * P2[3, 3]
        
        V2 = np.dot(ALPHA2, np.dot(P2, ALPHA2.T)) + K2

        V2 = (V2 + V2.T) / 2
        V2 = make_positive_definite(V2)

        if not allclose(V2, V2.T):
            raise ValueError("Covariance matrix V2 is not symmetric")

        if not np.all(np.linalg.eigvals(V2) > 0):
            raise ValueError("Covariance matrix V2 is not positive definite")
         
        V2INV = np.linalg.inv(V2)
            
        F2 = multivariate_normal_pdf(Y2[I, :22], mean=ALPHA2 @ A2, cov=V2)

        A3 = (np.dot(G2, A2) + np.dot(G2, np.dot(P2, np.dot(ALPHA2.T, np.dot(V2INV, (Y2[I, :22] - np.dot(ALPHA2, A2)))))) + B2 @ MEDU[I,:])

        P3 = (np.dot(G2, np.dot(P2, G2.T)) - np.dot(G2, np.dot(P2, np.dot(ALPHA2.T, np.dot(V2INV, np.dot(ALPHA2, np.dot(P2, G2.T)))))) + Q2)

        K3 = np.zeros((19, 19))
        K3[0, 0] = COV_ALL[6, 6] - P3[0, 0]
        K3[1, 1] = COV_ALL[7, 7] - (ALPHA3[1, 0] ** 2) * P3[0, 0]
        K3[2, 2] = COV_ALL[8, 8] - (ALPHA3[2, 0] ** 2) * P3[0, 0]
        K3[3, 3] = COV_ALL[13, 13] - P3[1, 1]
        K3[4, 4] = COV_ALL[14, 14] - (ALPHA3[4, 1] ** 2) * P3[1, 1]
        K3[5, 5] = COV_ALL[25, 25] - P3[2, 2]
        K3[6, 6] = COV_ALL[26, 26] - (ALPHA3[6, 2] ** 2) * P3[2, 2]
        K3[7, 7] = COV_ALL[27, 27] - (ALPHA3[7, 2] ** 2) * P3[2, 2]
        K3[8, 8] = COV_ALL[56, 56] - P3[3, 3]
        K3[9, 9] = COV_ALL[57, 57] - (ALPHA3[9, 3] ** 2) * P3[3, 3]
        K3[10, 10] = COV_ALL[58, 58] - (ALPHA3[10, 3] ** 2) * P3[3, 3]
        K3[11, 11] = COV_ALL[59, 59] - (ALPHA3[11, 3] ** 2) * P3[3, 3]
        K3[12, 12] = COV_ALL[60, 60] - (ALPHA3[12, 3] ** 2) * P3[3, 3]
        K3[13, 13] = COV_ALL[61, 61] - (ALPHA3[13, 3] ** 2) * P3[3, 3]
        K3[14, 14] = COV_ALL[62, 62] - (ALPHA3[14, 3] ** 2) * P3[3, 3]
        K3[15, 15] = COV_ALL[63, 63] - (ALPHA3[15, 3] ** 2) * P3[3, 3]
        K3[16, 16] = COV_ALL[64, 64] - (ALPHA3[16, 3] ** 2) * P3[3, 3]
        K3[17, 17] = COV_ALL[65, 65] - (ALPHA3[17, 3] ** 2) * P3[3, 3]
        K3[18, 18] = COV_ALL[66, 66] - (ALPHA3[18, 3] ** 2) * P3[3, 3]
        
        V3 = np.dot(ALPHA3, np.dot(P3, ALPHA3.T)) + K3

        V3 = (V3 + V3.T) / 2
        V3 = make_positive_definite(V3)

        if not allclose(V3, V3.T):
            raise ValueError("Covariance matrix V3 is not symmetric")

        if not np.all(np.linalg.eigvals(V3) > 0):
            raise ValueError("Covariance matrix V3 is not positive definite")
            
        F3 = multivariate_normal_pdf(Y3[I, :19], mean=ALPHA3 @ A3, cov=V3)
        
        L[I] = F1 * F2 * F3

    F4 = np.sum(np.log(L[L != 0]))
    FU = -F4 / NS

    return FU
