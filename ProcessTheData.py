
# coding: utf-8

# In[10]:

def ProcessTheData(filename,miss_threshold=0,corr_threshold=None):
    
    
    #Importing required packages

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import skew
    from IPython.display import display
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    import warnings
    
    warnings.filterwarnings('ignore')
    
    
    # Read data and obtain coefficients
    train = pd.read_csv(filename) #input('Enter the data path:'))
    
    #miss_threshold=float(input('Threshold for missing data (%):'))
    #corr_threshold=float(input('Threshold for correlation (0.0 - 1.0):'))
    
    #Removing duplicates
    train=train.drop_duplicates(keep='first', inplace=False)
    
    # Looking for outliers, as indicated by the author of the dataset
    train = train[train.GrLivArea < 4000]
    
    # Log transform the target for official scoring
    train.SalePrice = np.log1p(train.SalePrice)
    
    #Removing features with more than x% missing data

    missing = pd.DataFrame(train.isnull().sum())
    missing.columns=['No_missing_values']
    missing['Percentage_of_data_missing']=(missing.No_missing_values/train.shape[0])*100
    features = missing[missing.Percentage_of_data_missing <= miss_threshold].index
    train=train[features] 
    
    # Splitting data into numerical and categorical features 

    categorical_features = train.select_dtypes(include = ["object"]).columns
    numerical_features = train.select_dtypes(exclude = ["object"]).columns

    train_num = train[numerical_features]
    train_cat = train[categorical_features]
    
    ####################Processing categorical data################################
    
    # Handle missing values for features where median/mean/mode doesn't make sense


    # BsmtQual etc : data description says NA for basement features is "no basement"
    train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
    train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
    train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
    train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
    train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")

    # CentralAir : NA most likely means No
    train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")

    # Condition : NA most likely means Normal
    train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
    train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")

    # External stuff : NA most likely means average
    train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
    train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")

    # FireplaceQu : data description says NA means "no fireplace"
    train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")

    # Functional : data description says NA means typical 
    train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")

    # GarageType etc : data description says NA for garage features is "no garage"
    train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
    train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
    train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
    train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")

    # HeatingQC : NA most likely means typical
    train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")

    # KitchenQual : NA most likely means typical
    train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")

    # LotShape : NA most likely means regular
    train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")

    # MasVnrType : NA most likely means no veneer
    train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")

    # PavedDrive : NA most likely means not paved
    train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")

    # SaleCondition : NA most likely means normal sale
    train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")

    # Utilities : NA most likely means all public utilities
    train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
    
    #One-hot encoding for categorical data
    
    train_cat = pd.get_dummies(train_cat)


    ####################Processing numerical data################################
    
    # Handle missing values for numerical features by using median as replacement
    train_num = train_num.fillna(train_num.median())

    # Log transform of the skewed numerical features to lessen impact of outliers
    # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    skewness = train_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    skewed_features = skewness.index
    train_num[skewed_features] = np.log1p(train_num[skewed_features])
    
    ####################Combining numerical and categorical data################################
    
    # Join categorical and numerical features
    train = pd.concat([train_num, train_cat], axis = 1)
    
    
    
    
    #Removing features with less than x correlation with the SalePrice

    corr = train.corr()
    corr_with_SalePrice=pd.DataFrame(corr.SalePrice)
    if corr_threshold!=None:
        corr_with_SalePrice=corr_with_SalePrice[abs(corr_with_SalePrice.SalePrice)>corr_threshold]
        
    corr_with_SalePrice.index
    features=corr_with_SalePrice.index
    features=features.insert(0,"Id")

    train=train[features]

    
    #Removing features with low variance
    
    Id=train['Id']
    SalePrice=train['SalePrice']
    train.drop("SalePrice", axis = 1, inplace = True)
    train.drop("Id", axis = 1, inplace = True)
    
    
    
    def VarianceThreshold_selector(data):

        #Select Model
        selector = VarianceThreshold(threshold=(.8 * (1 - .8)))

        #Fit the Model
        selector.fit(data)
        features = selector.get_support(indices = True) #returns an array of integers corresponding to nonremoved features
        Features = list(data)
        features = [Features[i] for i in features] #Array of all nonremoved features' names
        #Format and Return
        selector = pd.DataFrame(selector.transform(data))
        selector.columns = features
        return selector


    train=VarianceThreshold_selector(train)
    
    train['Id']=Id
    train['SalePrice']=SalePrice
    
    return train

