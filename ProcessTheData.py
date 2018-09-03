def ProcessTheData(filename,miss_threshold=0,corr_threshold=None,final_features=None,catnum_features=None):
    
    
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

    

    
    if 'train.csv' in filename:
        
        # Looking for outliers, as indicated by the author of the dataset
        train = train[train.GrLivArea < 4000]
            
        # Log transform the target for official scoring
        train.SalePrice = np.log1p(train.SalePrice)
        
        #Removing features with more than x% missing data

        missing = pd.DataFrame(train.isnull().sum())
        missing.columns=['No_missing_values']
        missing['Percentage_of_data_missing']=(missing.No_missing_values/train.shape[0])*100
        missing = missing[missing.Percentage_of_data_missing <= miss_threshold]
        train=train[missing.index]

    # Splitting data into numerical and categorical features 

        categorical_features = train.select_dtypes(include = ["object"]).columns
        numerical_features = train.select_dtypes(exclude = ["object"]).columns


        train_num = train[numerical_features]
        train_cat = train[categorical_features]
        CatNum_features = [categorical_features, numerical_features]

        ####################Processing categorical data################################

        # Handle missing values for features where median/mean/mode doesn't make sense


        # BsmtQual etc : data description says NA for basement features is "no basement"
        train_cat.loc[:, "BsmtQual"] = train_cat.loc[:, "BsmtQual"].fillna("No")
        train_cat.loc[:, "BsmtCond"] = train_cat.loc[:, "BsmtCond"].fillna("No")
        train_cat.loc[:, "BsmtExposure"] = train_cat.loc[:, "BsmtExposure"].fillna("No")
        train_cat.loc[:, "BsmtFinType1"] = train_cat.loc[:, "BsmtFinType1"].fillna("No")
        train_cat.loc[:, "BsmtFinType2"] = train_cat.loc[:, "BsmtFinType2"].fillna("No")

        # CentralAir : NA most likely means No
        train_cat.loc[:, "CentralAir"] = train_cat.loc[:, "CentralAir"].fillna("N")

        # Condition : NA most likely means Normal
        train_cat.loc[:, "Condition1"] = train_cat.loc[:, "Condition1"].fillna("Norm")
        train_cat.loc[:, "Condition2"] = train_cat.loc[:, "Condition2"].fillna("Norm")

        # External stuff : NA most likely means average
        train_cat.loc[:, "ExterCond"] = train_cat.loc[:, "ExterCond"].fillna("TA")
        train_cat.loc[:, "ExterQual"] = train_cat.loc[:, "ExterQual"].fillna("TA")

        # FireplaceQu : data description says NA means "no fireplace"
        train_cat.loc[:, "FireplaceQu"] = train_cat.loc[:, "FireplaceQu"].fillna("No")

        # Functional : data description says NA means typical 
        train_cat.loc[:, "Functional"] = train_cat.loc[:, "Functional"].fillna("Typ")

        # GarageType etc : data description says NA for garage features is "no garage"
        train_cat.loc[:, "GarageType"] = train_cat.loc[:, "GarageType"].fillna("No")
        train_cat.loc[:, "GarageFinish"] = train_cat.loc[:, "GarageFinish"].fillna("No")
        train_cat.loc[:, "GarageQual"] = train_cat.loc[:, "GarageQual"].fillna("No")
        train_cat.loc[:, "GarageCond"] = train_cat.loc[:, "GarageCond"].fillna("No")

        # HeatingQC : NA most likely means typical
        train_cat.loc[:, "HeatingQC"] = train_cat.loc[:, "HeatingQC"].fillna("TA")

        # KitchenQual : NA most likely means typical
        train_cat.loc[:, "KitchenQual"] = train_cat.loc[:, "KitchenQual"].fillna("TA")

        # LotShape : NA most likely means regular
        train_cat.loc[:, "LotShape"] = train_cat.loc[:, "LotShape"].fillna("Reg")

        # MasVnrType : NA most likely means no veneer
        train_cat.loc[:, "MasVnrType"] = train_cat.loc[:, "MasVnrType"].fillna("None")

        # PavedDrive : NA most likely means not paved
        train_cat.loc[:, "PavedDrive"] = train_cat.loc[:, "PavedDrive"].fillna("N")

        # SaleCondition : NA most likely means normal sale
        train_cat.loc[:, "SaleCondition"] = train_cat.loc[:, "SaleCondition"].fillna("Normal")

        # Utilities : NA most likely means all public utilities
        train_cat.loc[:, "Utilities"] = train_cat.loc[:, "Utilities"].fillna("AllPub")



        #One-hot encoding for categorical data

        train_cat = pd.get_dummies(train_cat)


        ####################Processing numerical data################################

        # Handle missing values for numerical features by using median as replacement
        train_num = train_num.fillna(train_num.median(),inplace=True)
        
       
    


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
        train.drop("Id", axis = 1, inplace = True)


        SalePrice=train['SalePrice']
        train.drop("SalePrice", axis = 1, inplace = True)



        def VarianceThreshold_selector(data):

            #Select Model
            selector = VarianceThreshold(threshold=(.8 * (1 - .8)))

            #Fit the Model
            selector.fit(data)
            features = selector.get_support(indices = True) #returns an array of integers corresponding to nonremoved features
            #print (features)
            Features = list(data)
            features = [Features[i] for i in features]
            #features = [column for column in data[features]] #Array of all nonremoved features' names
            #print (features)
            #Format and Return
            selector = pd.DataFrame(selector.transform(data))
            selector.columns = features
            return selector




        train=VarianceThreshold_selector(train)
        train['Id']=Id
        final_features=train.columns
        train['SalePrice']=SalePrice

        train=train[~train.SalePrice.isnull()]


        return train,CatNum_features, final_features

    else:
        #catnum_features=CatNum_features


        # Splitting data into numerical and categorical features 
        
        if catnum_features!=None:
            categorical_features = catnum_features[0]
            numerical_features = catnum_features[1]
            numerical_features=numerical_features.drop('SalePrice')
            train_num=train[numerical_features]
            train_cat = train[categorical_features]


        
        ####################Processing categorical data################################

        # Handle missing values for features where median/mean/mode doesn't make sense


        # BsmtQual etc : data description says NA for basement features is "no basement"
        train_cat.loc[:, "BsmtQual"] = train_cat.loc[:, "BsmtQual"].fillna("No")
        train_cat.loc[:, "BsmtCond"] = train_cat.loc[:, "BsmtCond"].fillna("No")
        train_cat.loc[:, "BsmtExposure"] = train_cat.loc[:, "BsmtExposure"].fillna("No")
        train_cat.loc[:, "BsmtFinType1"] = train_cat.loc[:, "BsmtFinType1"].fillna("No")
        train_cat.loc[:, "BsmtFinType2"] = train_cat.loc[:, "BsmtFinType2"].fillna("No")

        # CentralAir : NA most likely means No
        train_cat.loc[:, "CentralAir"] = train_cat.loc[:, "CentralAir"].fillna("N")

        # Condition : NA most likely means Normal
        train_cat.loc[:, "Condition1"] = train_cat.loc[:, "Condition1"].fillna("Norm")
        train_cat.loc[:, "Condition2"] = train_cat.loc[:, "Condition2"].fillna("Norm")

        # External stuff : NA most likely means average
        train_cat.loc[:, "ExterCond"] = train_cat.loc[:, "ExterCond"].fillna("TA")
        train_cat.loc[:, "ExterQual"] = train_cat.loc[:, "ExterQual"].fillna("TA")

        # FireplaceQu : data description says NA means "no fireplace"
        train_cat.loc[:, "FireplaceQu"] = train_cat.loc[:, "FireplaceQu"].fillna("No")

        # Functional : data description says NA means typical 
        train_cat.loc[:, "Functional"] = train_cat.loc[:, "Functional"].fillna("Typ")

        # GarageType etc : data description says NA for garage features is "no garage"
        train_cat.loc[:, "GarageType"] = train_cat.loc[:, "GarageType"].fillna("No")
        train_cat.loc[:, "GarageFinish"] = train_cat.loc[:, "GarageFinish"].fillna("No")
        train_cat.loc[:, "GarageQual"] = train_cat.loc[:, "GarageQual"].fillna("No")
        train_cat.loc[:, "GarageCond"] = train_cat.loc[:, "GarageCond"].fillna("No")

        # HeatingQC : NA most likely means typical
        train_cat.loc[:, "HeatingQC"] = train_cat.loc[:, "HeatingQC"].fillna("TA")

        # KitchenQual : NA most likely means typical
        train_cat.loc[:, "KitchenQual"] = train_cat.loc[:, "KitchenQual"].fillna("TA")

        # LotShape : NA most likely means regular
        train_cat.loc[:, "LotShape"] = train_cat.loc[:, "LotShape"].fillna("Reg")

        # MasVnrType : NA most likely means no veneer
        train_cat.loc[:, "MasVnrType"] = train_cat.loc[:, "MasVnrType"].fillna("None")

        # PavedDrive : NA most likely means not paved
        train_cat.loc[:, "PavedDrive"] = train_cat.loc[:, "PavedDrive"].fillna("N")

        # SaleCondition : NA most likely means normal sale
        train_cat.loc[:, "SaleCondition"] = train_cat.loc[:, "SaleCondition"].fillna("Normal")

        # Utilities : NA most likely means all public utilities
        train_cat.loc[:, "Utilities"] = train_cat.loc[:, "Utilities"].fillna("AllPub")

        #One-hot encoding for categorical data

        train_cat = pd.get_dummies(train_cat)


        ####################Processing numerical data################################

        # Handle missing values for numerical features by using median as replacement
        train_num = train_num.fillna(train_num.median())
        

        # Standardize numerical features
        stdSc = StandardScaler()
        #train_num = pd.DataFrame(stdSc.fit_transform(train_num), index=train_num.index, columns=train_num.columns)


        # Log transform of the skewed numerical features to lessen impact of outliers
        # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
        skewness = train_num.apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > 0.5]
        skewed_features = skewness.index
        train_num[skewed_features] = np.log1p(train_num[skewed_features])

        ####################Combining numerical and categorical data################################

        # Join categorical and numerical features
        train = pd.concat([train_num, train_cat], axis = 1)

        train=train[final_features]

    
        
        
        return train



