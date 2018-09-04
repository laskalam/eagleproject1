import sys
import numpy as np
import pandas as pd
import seaborn as sns

    
def ProcessTheData(train_path, test_path, GrLivAreaLim=4000,SalePriceLim=500000,miss_threshold=50.0,corr_threshold=0.5):
    
    
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


    # Read data
    sys.stdout.write('Training data: %s\n Testing data: %s\n'
                     %(train_path,test_path))
    #Load the data
    train = pd.read_csv(str(train_path))
    test = pd.read_csv(str(test_path))
    
    

    #miss_threshold=float(input('Threshold for missing data (%):'))
    #corr_threshold=float(input('Threshold for correlation (0.0 - 1.0):'))

    #Removing duplicates
    train=train.drop_duplicates(keep='first', inplace=False)

    # separate ID from the features
    train_ID = train['Id']
    test_ID = test['Id']
     
    train.drop('Id', axis = 1, inplace = True)
    test.drop('Id', axis = 1, inplace = True)

  
    #display_distrib(train, 'SalePrice')
    
    sys.stdout.write('Train data shape: %d, %d\n'%(train.shape))
    sys.stdout.write('Test data shape: %d, %d\n'%(test.shape))
    
#Preprocessing training data

    # Looking for outliers, as indicated by the author of the dataset
    # analyze and remove huge outliers: GrLivArea, ...
    #display_outlier(train, 'GrLivArea')
    i_ =  train[(train['GrLivArea']>GrLivAreaLim) &
                              (train['SalePrice']>SalePriceLim)].index
    train = train.drop(i_)
    
    # Log transform the target for official scoring
    train.SalePrice = np.log1p(train.SalePrice)

    y_train = train.SalePrice.values
  
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

    #Processing categorical data#

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


    #Processing numerical data#
    
    # Handle missing values for numerical features by using median as replacement
    #train_num = pd.DataFrame(train_num.fillna(train_num.median(),inplace=True))
    train_num = train_num.fillna(train_num.median())

    # Log transform of the skewed numerical features to lessen impact of outliers
    # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    skewness = train_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    skewed_features = skewness.index
    train_num[skewed_features] = np.log1p(train_num[skewed_features])

    #Removing features with less than x correlation with the SalePrice

    corr = train_num.corr()

    corr_with_SalePrice=pd.DataFrame(corr.SalePrice)
    if corr_threshold!=None:
        corr_with_SalePrice=corr_with_SalePrice[abs(corr_with_SalePrice.SalePrice)>corr_threshold]

    corr_with_SalePrice.index
    features=corr_with_SalePrice.index

    train_num=train_num[features]

    # Join categorical and numerical features
    train = pd.concat([train_num, train_cat], axis = 1)







    #Removing features with low variance


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
    final_features=train.columns
    train['SalePrice']=y_train

    y_train=y_train[~train.SalePrice.isnull()]
    train=train[~train.SalePrice.isnull()]

    train=train.drop('SalePrice',axis=1)
    

#Preprocessing test data


    # Splitting data into numerical and categorical features 

    
    categorical_features = CatNum_features[0]
    numerical_features = CatNum_features[1]
    numerical_features=numerical_features.drop('SalePrice')
    test_num=test[numerical_features]
    test_cat = test[categorical_features]



    ##Processing categorical data#

    # Handle missing values for features where median/mean/mode doesn't make sense

    # BsmtQual etc : data description says NA for basement features is "no basement"
    test_cat.loc[:, "BsmtQual"] = test_cat.loc[:, "BsmtQual"].fillna("No")
    test_cat.loc[:, "BsmtCond"] = test_cat.loc[:, "BsmtCond"].fillna("No")
    test_cat.loc[:, "BsmtExposure"] = test_cat.loc[:, "BsmtExposure"].fillna("No")
    test_cat.loc[:, "BsmtFinType1"] = test_cat.loc[:, "BsmtFinType1"].fillna("No")
    test_cat.loc[:, "BsmtFinType2"] = test_cat.loc[:, "BsmtFinType2"].fillna("No")

    # CentralAir : NA most likely means No
    test_cat.loc[:, "CentralAir"] = test_cat.loc[:, "CentralAir"].fillna("N")

    # Condition : NA most likely means Normal
    test_cat.loc[:, "Condition1"] = test_cat.loc[:, "Condition1"].fillna("Norm")
    test_cat.loc[:, "Condition2"] = test_cat.loc[:, "Condition2"].fillna("Norm")

    # External stuff : NA most likely means average
    test_cat.loc[:, "ExterCond"] = test_cat.loc[:, "ExterCond"].fillna("TA")
    test_cat.loc[:, "ExterQual"] = test_cat.loc[:, "ExterQual"].fillna("TA")

    # FireplaceQu : data description says NA means "no fireplace"
    test_cat.loc[:, "FireplaceQu"] = test_cat.loc[:, "FireplaceQu"].fillna("No")

    # Functional : data description says NA means typical 
    test_cat.loc[:, "Functional"] = test_cat.loc[:, "Functional"].fillna("Typ")

    # GarageType etc : data description says NA for garage features is "no garage"
    test_cat.loc[:, "GarageType"] = test_cat.loc[:, "GarageType"].fillna("No")
    test_cat.loc[:, "GarageFinish"] = test_cat.loc[:, "GarageFinish"].fillna("No")
    test_cat.loc[:, "GarageQual"] = test_cat.loc[:, "GarageQual"].fillna("No")
    test_cat.loc[:, "GarageCond"] = test_cat.loc[:, "GarageCond"].fillna("No")

    # HeatingQC : NA most likely means typical
    test_cat.loc[:, "HeatingQC"] = test_cat.loc[:, "HeatingQC"].fillna("TA")

    # KitchenQual : NA most likely means typical
    test_cat.loc[:, "KitchenQual"] = test_cat.loc[:, "KitchenQual"].fillna("TA")

    # LotShape : NA most likely means regular
    test_cat.loc[:, "LotShape"] = test_cat.loc[:, "LotShape"].fillna("Reg")

    # MasVnrType : NA most likely means no veneer
    test_cat.loc[:, "MasVnrType"] = test_cat.loc[:, "MasVnrType"].fillna("None")

    # PavedDrive : NA most likely means not paved
    test_cat.loc[:, "PavedDrive"] = test_cat.loc[:, "PavedDrive"].fillna("N")

    # SaleCondition : NA most likely means normal sale
    test_cat.loc[:, "SaleCondition"] = test_cat.loc[:, "SaleCondition"].fillna("Normal")

    # Utilities : NA most likely means all public utilities
    test_cat.loc[:, "Utilities"] = test_cat.loc[:, "Utilities"].fillna("AllPub")

    #One-hot encoding for categorical data

    test_cat = pd.get_dummies(test_cat)


    ##Processing numerical data##

    # Handle missing values for numerical features by using median as replacement
    test_num = test_num.fillna(test_num.median())
    #print(type(test_num))
    test_num=pd.DataFrame(test_num)
    #print(type(test_num))
    
    # Log transform of the skewed numerical features to lessen impact of outliers
    # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    skewness = test_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    skewed_features = skewness.index
    test_num[skewed_features] = np.log1p(test_num[skewed_features])

    ##Combining numerical and categorical data#

    # Join categorical and numerical features
    test = pd.concat([test_num, test_cat], axis = 1)

    test=test[final_features]




    return train,test, y_train, train_ID, test_ID





