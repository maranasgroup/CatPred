import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import ipdb

res = np.array(pd.read_csv('catpred_kcat/all_samples_catpredSplits.csv')).T
sequence = res[1]
smiles = res[2]
Type = res[6]
Value = res[7]
Predict_Label = res[8]
Training_test = res[9]
print(sequence[0], smiles[0], Type[0], Value[0], Predict_Label[0], Training_test[0])

OOD_INDICES = [3,
 15,
 28,
 36,
 45,
 48,
 50,
 71,
 82,
 83,
 85,
 86,
 92,
 107,
 109,
 131,
 134,
 135,
 142,
 144,
 146,
 147,
 163,
 173,
 180,
 185,
 187,
 203,
 205,
 211,
 212,
 220,
 228,
 233,
 235,
 236,
 242,
 247,
 257,
 260,
 266,
 276,
 297,
 301,
 302,
 306,
 313,
 314,
 327,
 331,
 343,
 346,
 350,
 356,
 382,
 388,
 395,
 399,
 402,
 403,
 413,
 427,
 434,
 435,
 437,
 438,
 447,
 454,
 466,
 470,
 497,
 503,
 508,
 510,
 511,
 513,
 514,
 518,
 526,
 538,
 541,
 543,
 563,
 564,
 565,
 567,
 568,
 572,
 580,
 590,
 591,
 607,
 610,
 627,
 633,
 636,
 640,
 642,
 656,
 660,
 668,
 690,
 697,
 705,
 706,
 716,
 726,
 728,
 741,
 742,
 746,
 754,
 759,
 769,
 770,
 776,
 778,
 783,
 788,
 789,
 795,
 797,
 800,
 803,
 810,
 815,
 816,
 832,
 840,
 844,
 848,
 849,
 873,
 874,
 877,
 879,
 882,
 891,
 903,
 904,
 908,
 914,
 916,
 926,
 938,
 941,
 949,
 961,
 963,
 970,
 973,
 985,
 990,
 996,
 1019,
 1020,
 1021,
 1022,
 1030,
 1032,
 1039,
 1050,
 1056,
 1072,
 1085,
 1089,
 1090,
 1100,
 1110,
 1112,
 1115,
 1132,
 1138,
 1149,
 1157,
 1199,
 1210,
 1213,
 1240,
 1248,
 1251,
 1271,
 1276,
 1280,
 1290,
 1292,
 1293,
 1309,
 1320,
 1331,
 1341,
 1346,
 1351,
 1353,
 1377,
 1380,
 1382,
 1388,
 1394,
 1395,
 1396,
 1408,
 1409,
 1421,
 1424,
 1430,
 1440,
 1443,
 1447,
 1461,
 1468,
 1469,
 1473,
 1474,
 1477,
 1478,
 1488,
 1494,
 1499,
 1502,
 1503,
 1520,
 1535,
 1538,
 1541,
 1551,
 1553,
 1560,
 1572,
 1573,
 1576,
 1580,
 1592,
 1595,
 1597,
 1610,
 1612,
 1614,
 1652,
 1661,
 1662,
 1671,
 1673,
 1675,
 1685,
 1688,
 1698,
 1704,
 1708,
 1724,
 1726,
 1737,
 1738,
 1739,
 1742,
 1749,
 1751,
 1756,
 1759,
 1760,
 1761,
 1762,
 1764,
 1768,
 1771,
 1777,
 1778,
 1788,
 1801,
 1804,
 1815,
 1817,
 1822,
 1837,
 1846,
 1847,
 1848,
 1849,
 1853,
 1855,
 1879,
 1882,
 1885,
 1893,
 1900,
 1908,
 1910,
 1913,
 1924,
 1930,
 1944,
 1965,
 1972,
 1974,
 1989,
 2004,
 2014,
 2016,
 2020,
 2035,
 2040,
 2041,
 2047,
 2062,
 2066,
 2080,
 2082,
 2083,
 2096,
 2097,
 2108,
 2115,
 2120,
 2124,
 2128,
 2131,
 2142,
 2156,
 2166,
 2167,
 2173,
 2174,
 2178,
 2191,
 2202,
 2204,
 2207,
 2219,
 2228,
 2231,
 2240,
 2241,
 2248,
 2252,
 2259,
 2265,
 2273,
 2278,
 2281,
 2301,
 2302,
 2303,
 2304,
 2314]

def Whole_dataset():
    # Calculate the whole dataset
    Pcc = pearsonr(Value, Predict_Label)[0]
    RMSE = np.sqrt(mean_squared_error(Value, Predict_Label))
    MAE = mean_absolute_error(Value, Predict_Label)
    r2 = r2_score(Value, Predict_Label)
    print('***The whole set***')
    print('Pcc:', Pcc, 'RMSE:', RMSE, 'MAE:', MAE, 'r2:', r2)


def test_dataset():
    # Calculate the test dataset
    Value_test = []
    Predict_Label_test = []
    for i in range(len(Training_test)):
        if Training_test[i] == 1:
            Value_test.append(Value[i])
            Predict_Label_test.append(Predict_Label[i])
    Value_test = np.array(Value_test)
    Predict_Label_test = np.array(Predict_Label_test)
    
    Value_test_ood = Value_test[OOD_INDICES]
    Predict_Label_test_ood = Predict_Label_test[OOD_INDICES]
    Pcc_test = pearsonr(Value_test, Predict_Label_test)[0]
    RMSE_test = np.sqrt(mean_squared_error(Value_test, Predict_Label_test))
    MAE_test = mean_absolute_error(Value_test, Predict_Label_test)
    r2_test = r2_score(Value_test, Predict_Label_test)
    
    # ipdb.set_trace()
    errors = np.abs(Value_test-Predict_Label_test)
    p1mag = len(errors[errors<1])/len(errors)
    
    print('mae_p1mag', np.average(errors))
          
    print('***Test set***')
    print('Pcc:', Pcc_test, 'RMSE:', RMSE_test, 'MAE:', MAE_test, 'r2:', r2_test, 'p1mag:', p1mag)
    
    Pcc_test = pearsonr(Value_test_ood, Predict_Label_test_ood)[0]
    RMSE_test = np.sqrt(mean_squared_error(Value_test_ood, Predict_Label_test_ood))
    MAE_test = mean_absolute_error(Value_test_ood, Predict_Label_test_ood)
    r2_test = r2_score(Value_test_ood, Predict_Label_test_ood)
    
    errors = np.abs(Value_test_ood-Predict_Label_test_ood)
    p1mag = len(errors[errors<1])/len(errors)
    
    print('mae_p1mag', np.average(errors))
          
    print('***Test set OOD***')
    print(Value_test_ood[0:5])
    print('Pcc:', Pcc_test, 'RMSE:', RMSE_test, 'MAE:', MAE_test, 'r2:', r2_test, 'p1mag:', p1mag)


def Wildtype_all_dataset():
    # Calculate the Wildtype/Mutant dataset
    Value_wildtype = []
    Predict_Label_wildtype = []
    for i in range(len(Type)):
        if Type[i] == 'wildtype':
            Value_wildtype.append(Value[i])
            Predict_Label_wildtype.append(Predict_Label[i])
    Value_wildtype = np.array(Value_wildtype)
    Predict_Label_wildtype = np.array(Predict_Label_wildtype)
    Pcc_test = pearsonr(Value_wildtype, Predict_Label_wildtype)[0]
    RMSE_test = np.sqrt(mean_squared_error(Value_wildtype, Predict_Label_wildtype))
    MAE_test = mean_absolute_error(Value_wildtype, Predict_Label_wildtype)
    r2_test = r2_score(Value_wildtype, Predict_Label_wildtype)
    print('***The whole wildtype set***')
    print('Pcc:', Pcc_test, 'RMSE:', RMSE_test, 'MAE:', MAE_test, 'r2:', r2_test)


def Wildtype_test_dataset():
    # Calculate the Wildtype/Mutant dataset
    Value_wildtype = []
    Predict_Label_wildtype = []
    for i in range(len(Type)):
        if Type[i] == 'wildtype' and Training_test[i] == 1:
            Value_wildtype.append(Value[i])
            Predict_Label_wildtype.append(Predict_Label[i])
    Value_wildtype = np.array(Value_wildtype)
    Predict_Label_wildtype = np.array(Predict_Label_wildtype)
    Pcc_test = pearsonr(Value_wildtype, Predict_Label_wildtype)[0]
    RMSE_test = np.sqrt(mean_squared_error(Value_wildtype, Predict_Label_wildtype))
    MAE_test = mean_absolute_error(Value_wildtype, Predict_Label_wildtype)
    r2_test = r2_score(Value_wildtype, Predict_Label_wildtype)
    print('***The test wildtype set***')
    print('Pcc:', Pcc_test, 'RMSE:', RMSE_test, 'MAE:', MAE_test, 'r2:', r2_test)


def Mutant_all_dataset():
    # Calculate the Wildtype/Mutant dataset
    Value_wildtype = []
    Predict_Label_wildtype = []
    for i in range(len(Type)):
        if Type[i] != 'wildtype':
            Value_wildtype.append(Value[i])
            Predict_Label_wildtype.append(Predict_Label[i])
    Value_wildtype = np.array(Value_wildtype)
    Predict_Label_wildtype = np.array(Predict_Label_wildtype)
    Pcc_test = pearsonr(Value_wildtype, Predict_Label_wildtype)[0]
    RMSE_test = np.sqrt(mean_squared_error(Value_wildtype, Predict_Label_wildtype))
    MAE_test = mean_absolute_error(Value_wildtype, Predict_Label_wildtype)
    r2_test = r2_score(Value_wildtype, Predict_Label_wildtype)
    print('***The whole mutant set***')
    print('Pcc:', Pcc_test, 'RMSE:', RMSE_test, 'MAE:', MAE_test, 'r2:', r2_test)


def Mutant_test_dataset():
    # Calculate the Wildtype/Mutant dataset
    Value_wildtype = []
    Predict_Label_wildtype = []
    for i in range(len(Type)):
        if Type[i] != 'wildtype' and Training_test[i] == 1:
            Value_wildtype.append(Value[i])
            Predict_Label_wildtype.append(Predict_Label[i])
    Value_wildtype = np.array(Value_wildtype)
    Predict_Label_wildtype = np.array(Predict_Label_wildtype)
    Pcc_test = pearsonr(Value_wildtype, Predict_Label_wildtype)[0]
    RMSE_test = np.sqrt(mean_squared_error(Value_wildtype, Predict_Label_wildtype))
    MAE_test = mean_absolute_error(Value_wildtype, Predict_Label_wildtype)
    r2_test = r2_score(Value_wildtype, Predict_Label_wildtype)
    print('***The test mutant set***')
    print('Pcc:', Pcc_test, 'RMSE:', RMSE_test, 'MAE:', MAE_test, 'r2:', r2_test)


def New_substrate_enzyme_dataset():
    # Calculate the test New_substrate_enzyme dataset
    Trainingset_seq_smiles = []
    for i in range(len(Training_test)):
        if Training_test[i] == 0:
            Trainingset_seq_smiles.append(sequence[i])
            Trainingset_seq_smiles.append(smiles[i])
    Value_test = []
    Predict_Label_test = []
    for i in range(len(Training_test)):
        if Training_test[i] == 1 and (sequence[i] not in Trainingset_seq_smiles or smiles[i] not in Trainingset_seq_smiles):
            Value_test.append(Value[i])
            Predict_Label_test.append(Predict_Label[i])
    Value_test = np.array(Value_test)
    Predict_Label_test = np.array(Predict_Label_test)
    Pcc_test = pearsonr(Value_test, Predict_Label_test)[0]
    RMSE_test = np.sqrt(mean_squared_error(Value_test, Predict_Label_test))
    MAE_test = mean_absolute_error(Value_test, Predict_Label_test)
    r2_test = r2_score(Value_test, Predict_Label_test)
    print('***The Test new_substrate_enzyme dataset***')
    print('Pcc:', Pcc_test, 'RMSE:', RMSE_test, 'MAE:', MAE_test, 'r2:', r2_test)
    
if __name__ == '__main__':
    Whole_dataset()
    test_dataset()
    # Wildtype_all_dataset()
    # Wildtype_test_dataset()
    # Mutant_all_dataset()
    # Mutant_test_dataset()
    # New_substrate_enzyme_dataset()
