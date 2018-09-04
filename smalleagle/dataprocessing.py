import sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def ProcessTheData(train_path, test_path, GrLivAreaLim=4000,
                   SalePriceLim=500000,miss_threshold=50.0,corr_threshold=0.5, TopMissingData=20):
#def ProcessTheData(train_path, test_path, GrLivAreaLim=4000,
#                   SalePriceLim=300000, TopMissingData=20):
    sys.stdout.write('Training data: %s\n Testing data: %s\n'
                     %(train_path,test_path))
    #Load the data
    train = pd.read_csv(str(train_path))
    test = pd.read_csv(str(test_path))
    
    
    # separate ID from the features
    train_ID = train['Id']
    test_ID = test['Id']
     
    train.drop('Id', axis = 1, inplace = True)
    test.drop('Id', axis = 1, inplace = True)
    
    # analyze and remove huge outliers: GrLivArea, ...
    #display_outlier(train, 'GrLivArea')
    train = train.drop(train[(train['GrLivArea']>GrLivAreaLim) &
                              (train['SalePrice']<SalePriceLim)].index)
    #display_outlier(train, 'GrLivArea')
    
    # normalize distribution of output (SalePrice)
    #display_distrib(train, 'SalePrice')
    train["SalePrice"] = np.log1p(train["SalePrice"])
    y_train = train.SalePrice.values
    #display_distrib(train, 'SalePrice')

    # Process the test and train sample together
    sys.stdout.write('Combine test/train for processing\n')
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train.drop(['SalePrice'], axis=1, inplace=True)
    all_data = pd.concat((train, test)).reset_index(drop=True)
    sys.stdout.write('Train data shape: %d, %d\n'%(train.shape))
    sys.stdout.write('Test data shape: %d, %d\n'%(test.shape))
    sys.stdout.write('Combined data shape: %d, %d\n'%(all_data.shape))

    # fill missing data
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:TopMissingData]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    sys.stdout.write('======================\n')
    sys.stdout.write('The discarded features are: Features & % of missing values\n')

    #for md in missing_data:
    #    sys.stdout.write('%s    '%md)
    sys.stdout.write('\n======================\n')

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None") #NA="No Pool".
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
                              #NA="no misc feature"
    all_data["Alley"] = all_data["Alley"].fillna("None")
                        #NA="no alley access"
    all_data["Fence"] = all_data["Fence"].fillna("None") #NA="no fence"
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
                              #NA="no fireplace"
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())) # fill by the median LotFrontage of all neighborhood because they have same lot frontage
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None') #NaN=no basement

    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
                             #NA=no masonry veneer
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
                             #NA=no masonry veneer
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
               #This feature is weard
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
                             #NA=typical
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0]) #Only one NA, give that the most frequent. 
    
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0]) #Similar situation with Electrical 
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) #Similar to Electrical
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0]) #Similar to Electrical
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0]) #Very few missing, the same method.
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None") #None might be appropriate.



    #Might be of interest
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] #Total Area of the house might be usefull.

    #this part needs a function, I am thinking of Tu.... transormation.
    for feature in all_data:
        if all_data[feature].dtype != "object":
            all_data[feature] = np.log1p(all_data[feature])



    #This is a test. Convert all numerical features into
    #categorical
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallQual'] = all_data['OverallQual'].astype(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)


    #Then categorical into dummies
    categorical_features = \
    ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
    'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
    'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
    'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallQual',
    'OverallCond', 'YrSold', 'MoSold')
    for c in categorical_features:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))
    sys.stdout.write('Data shape prior to dummies: %d, %d\n'%all_data.shape)
    #Make dummies
    all_data = pd.get_dummies(all_data)
    sys.stdout.write('Data shape after dummies: %d, %d\n'%all_data.shape)
   
    #Correlation cleaning 
    train = all_data[:ntrain]
    train['SalePrice'] = y_train
    corr = train.corr()
    corr_with_SalePrice=pd.DataFrame(corr.SalePrice)
    if corr_threshold!=None:
        corr_with_SalePrice=corr_with_SalePrice[abs(corr_with_SalePrice.SalePrice)>corr_threshold]
    corr_with_SalePrice.index
    features=corr_with_SalePrice.index
    all_data = all_data[features.drop('SalePrice')]

 
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    return train,test, y_train, train_ID, test_ID
