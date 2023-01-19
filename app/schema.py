from pydantic import BaseModel
from typing import Optional

class predictions(BaseModel):
    prediction: Optional[float]=0.0
    accuracy: Optional[float]=0.0

class Input_Features(BaseModel):
    
    Id: Optional[int] = 1
    MSSubClass: Optional[int] = 60
    LotArea: Optional[int] = 8450 
    OverallQual: Optional[int] = 7
    OverallCond: Optional[int] = 5
    YearBuilt: Optional[int] = 2003 
    YearRemodAdd: Optional[int] = 2003
    BsmtFinSF1: Optional[int] = 706
    BsmtFinSF2: Optional[int] = 0
    BsmtUnfSF: Optional[int] = 150 
    TotalBsmtSF: Optional[int] = 856
    FirstFlrSF: Optional[int] = 856
    SecondFlrSF: Optional[int] = 854
    LowQualFinSF: Optional[int] = 0
    GrLivArea: Optional[int] = 1710
    BsmtFullBath: Optional[int] = 1
    BsmtHalfBath: Optional[int] = 0
    FullBath: Optional[int] = 2
    HalfBath: Optional[int] = 1
    BedroomAbvGr: Optional[int] = 3
    KitchenAbvGr: Optional[int] = 1
    TotRmsAbvGrd: Optional[int] = 8
    Fireplaces: Optional[int] = 0
    GarageCars: Optional[int] = 2
    GarageArea: Optional[int] = 548
    WoodDeckSF: Optional[int] = 0
    OpenPorchSF: Optional[int] = 61
    EnclosedPorch: Optional[int] = 0
    ThirdSsnPorch: Optional[int] = 0
    ScreenPorch: Optional[int] = 0
    PoolArea: Optional[int] = 0 
    MiscVal: Optional[int] = 0
    MoSold: Optional[int] = 2
    YrSold: Optional[int] = 2008
    LotFrontage: Optional[float]=65.0
    MasVnrArea: Optional[float] = 196.0
    GarageYrBlt: Optional[float] = 2003.0
    MSZoning: Optional[str]="RL"
    Street: Optional[str] = "Pave"
    Alley: Optional[str] = "NA"
    LotShape: Optional[str] = "Reg"
    LandContour: Optional[str] = "Lvl"
    Utilities: Optional[str] = "AllPub"
    LotConfig: Optional[str] = "Inside"
    LandSlope: Optional[str] = "Gtl"
    Neighborhood: Optional[str] = "CollgCr"
    Condition1: Optional[str] = "Norm"
    Condition2: Optional[str] = "Norm"
    BldgType: Optional[str] = "1Fam"
    HouseStyle: Optional[str] = "2Story"
    RoofStyle: Optional[str] = "Gable"
    RoofMatl: Optional[str] = "CompShg"
    Exterior1st: Optional[str] = "VinylSd"
    Exterior2nd: Optional[str] = "VinylSd"
    MasVnrType: Optional[str] = "BrkFace"
    ExterQual: Optional[str] = "Gd"
    ExterCond: Optional[str] = "TA"
    Foundation: Optional[str] = "PConc"
    BsmtQual: Optional[str] = "Gd"
    BsmtCond: Optional[str] = "TA"
    BsmtExposure: Optional[str] = "No"
    BsmtFinType1: Optional[str] = "GLQ"
    BsmtFinType2: Optional[str] = "Unf"
    Heating: Optional[str] = "GasA"
    HeatingQC: Optional[str] = "Ex"
    CentralAir: Optional[str] = "Y"
    Electrical: Optional[str] = "SBrkr"
    KitchenQual: Optional[str] = "Gd"
    Functional: Optional[str] = "Typ"
    FireplaceQu: Optional[str] = "NA"
    GarageType: Optional[str] = "Attchd"
    GarageFinish: Optional[str] = "RFn"
    GarageQual: Optional[str] = "TA"
    GarageCond: Optional[str] = "TA"
    PavedDrive: Optional[str] = "Y"
    PoolQC: Optional[str] = "NA"
    Fence: Optional[str] = "NA"
    MiscFeature: Optional[str] = "NA"
    SaleType: Optional[str] = "WD"
    SaleCondition: Optional[str] = "Normal"    


