import warnings
# import numpy as np
import pandas as pd
from FeatureExtractor import *
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

columns_to_drop = ["MinBoxArea", "MaxBoxArea", "MinDistance", "MaxDistance", 'Black Box Frame Number',
                   'Black Box Filename', "Alert Type",
                   'TimeStamp', 'passed', 'Filename', 'alert_type', 'Frame_Number', 'Id', 'Class',
                   'ClassifierResult', 'Angle', "ClassifierResultWithRules", "AbsSpeedKMH", "TiltAngle",
                   "BoxLeft", "BoxRight", "MinLeft", "MinRight", "MaxLeft", "MaxRight", 'BoxLeftsFit.Intercept',
                   'BoxLeftsFit.Slope', 'BoxRightsFit.Intercept', 'BoxRightsFit.Slope', 'BoxWidth', 'BoxWidthPrev',
                   'BoxBottomPrev', 'BoxCenter', 'BoxCenterPrev', "DistanceToSideOuterPrev", "DistanceToSideInnerPrev"]

AlertType = 'Front_Collision'  # 'Front_Collision'  # 'BlindSpot' #'Side_Collision'  # 'Safe_Distance'
ModelFileName = AlertType.replace("_", "") + 'Model.json'
DataFileName = AlertType.replace("_", "") + ".csv"
TrainFileName = AlertType.replace("_", "") + "_Train.csv"
TestFileName = AlertType.replace("_", "") + "_Test.csv"
alert_string = "blind"
videos_directory = r"D:\Camera Roll\Alerts\Blindspot"
df = pd.read_csv(DataFileName)

if AlertType == 'BlindSpots' or AlertType == 'BlindSpot':
    AlertType = 'BlindSpot'  # 'Safe_Distance'
    videos_directory = r"D:\Camera Roll\Alerts\Blindspot"
    alert_string = "blind"
    df["MaxMinAngle"] = df["MaxAngleToCloserSide"] - df["MinAngleToCloserSide"]
    df["MaxMinAngleCloserSide"] = df["MaxAngle"] - df["MinAngle"]

    df["AngleMinusMin"] = df["AbsAngle"] - df["MinAngle"]
    df["MaxMinusAngle"] = df["MaxAngle"] - df["AbsAngle"]

    df["AngleToCloserSideMinusMin"] = abs(df["AngleToCloserSide"]) - df["MinAngleToCloserSide"]
    df["MaxMinusAngleToCloserSide"] = df["MaxAngleToCloserSide"] - abs(df["AngleToCloserSide"])
    df["MovmentAngleTan"] = df["HorizontalMovement"] / (df["VerticalMovement"] + 0.0000001)
    df["PredictedAngle"] = df["AbsAnglesFit.Intercept"] + 25 * df["AbsAnglesFit.Slope"]
    df["PredictedAngleToCloserSide"] = df["AbsAnglesToCloserSideFit.Intercept"] + 25 * df[
        "AbsAnglesToCloserSideFit.Slope"]

    # df["RelSpeedKMHOverDistanceToSideOuterBox"] = df["RelSpeedKMH"] / (df["DistanceToSideOuterBox"] + 0.0000000001)
    # df["RelSpeedKMHOverRotation"] = df["RelSpeedKMH"] / (df["Rotation"] + 0.0000000001)
    # df["RelSpeedKMHOverXInCollision"] = df["RelSpeedKMH"] / (df["DistanceXInCollision"] + 0.0000000001)
    # df["AbsAngleOverHorizontalMovement"] = df["AbsAngle"] / (df["HorizontalMovement"] + 0.0000000001)
    # df["AbsAngleOverBoxTop"] = df["AbsAngle"] / (df["BoxTop"] + 0.0000000001)
    # df["AbsAngleOverPredictedX"] = df["AbsAngle"] / (df["PredictedX"] + 0.0000000001)
    # df["AbsAngleOverYFitIntercept"] = df["AbsAngle"] / (df["YFit.Intercept"] + 0.0000000001)
    # df["DistanceOverAngleFromTop"] = df["Distance"] / (df["AngleFromTop"] + 0.0000000001)
    # df["DistanceOverMedianAngularVelocity"] = df["Distance"] / (df["AbsMedianAngularVelocity"] + 0.0000000001)
    # df["DistanceXOverMedianAngularVelocity"] = df["DistanceX"] / (df["AbsMedianAngularVelocity"] + 0.0000000001)
    # df["DistanceXOverMedianAngularVelocity"] = df["DistanceX"] / (df["AbsMedianAngularVelocity"] + 0.0000000001)
    # df["DistanceToSideOuterOverAngleFromTop"] = df["DistanceToSideOuterBox"] / (df["AngleFromTop"] + 0.0000000001)
    # df["DistanceToSideOuterOverDistanceFromClosestCorner"] = df["DistanceToSideOuterBox"] / (df["DistanceFromClosestCorner"] + 0.0000000001)
    # df["MovementRatio"] = df["HorizontalMovement"] / (df["BoxBottomMovement8Frames"] + 0.0000000001)
    #
    # df["MaxDistanceOverTimeToCollisionBoxFullSeries"] = df["MaxDistance"] / (df["TimeToCollisionBoxAreaFullSeries"] + 0.0000000001)

else:
    columns_to_drop = columns_to_drop + ["MinBoxBottom", "MaxBoxBottom", "MinAngle", "MaxAngle"]
    alert_string = "front"

    if AlertType == 'Safe_Distance':
        ModelFileName = 'SafeDistanceModel.json'
        alert_string = "safe"
    if AlertType == 'Side_Collision':
        AlertType = 'Front_Collision_Side'
        DataFileName = "FrontCollision.csv"

    videos_directory = r"D:\Camera Roll\Alerts\Front Alerts"

# df = add_features(df)
df = df.drop_duplicates()
df.replace('#DIV/0!', 0, inplace=True)
df.replace('#NAME?', 0, inplace=True)
df.replace('#REF!', 0, inplace=True)
df.replace('inf', 1000000, inplace=True)
# print(check_accuracy(df, videos_directory, alert_string))

Data = df.loc[df["Alert Type"] == AlertType]

# DataSD = DataSD.loc[df.Rotation == 1.571]
cols = list(Data.columns.values)
cols.pop(cols.index('label'))
Data = Data[cols + ['label']]

Data = Data[Data['label'] != -1]
Data = Data[Data['label'] != -2]

if 'passed' in Data.columns:
    Data = Data[Data['passed'] == 1]

Data["Car"] = np.where(Data['Class'] == 0, 1.0, 0.0)
Data["Bus"] = np.where(Data['Class'] == 1, 1.0, 0.0)
Data["Truck"] = np.where(Data['Class'] == 2, 1.0, 0.0)
Data["MotorCycle"] = np.where(Data['Class'] == 3, 1.0, 0.0)
# Data["InTrafficByCars"] = np.where(Data['NCars'] > 3, 1.0, 0.0)
if 'NCars' in Data.columns:
    del (Data['NCars'])
Data = Data[Data["ClassifierResultWithRules"] >= 0]
Data_drop = Data.copy()
for col in columns_to_drop:
    if col in Data_drop.columns:
        del (Data_drop[col])
for i in range(8):
    if 'BoxWidths[' + str(i) + ']' in Data_drop.columns:
        del (Data_drop['BoxWidths[' + str(i) + ']'])
        del (Data_drop['TimeDeltas[' + str(i) + ']'])
        del (Data_drop['BoxHeights[' + str(i) + ']'])
        del (Data_drop['BoxCenters[' + str(i) + ']'])
        del (Data_drop['BoxBottoms[' + str(i) + ']'])
        del (Data_drop['Angles[' + str(i) + ']'])
        del (Data_drop['Distances[' + str(i) + ']'])

if AlertType != 'BlindSpot':
    for col in ['MaxBoxArea', 'MinBoxArea', 'MaxBoxBottom', 'MinBoxBottom', 'MaxAngle', 'MinAngle', 'MaxDistance',
                'MinDistance', 'Counter', 'MovementAngle', 'MovementRadius', 'MaxDistanceToSideOuter',
                'MaxDistanceToSideInner', "HeightOverWidthIntercept", "MaxAngleToCloserSide", "MinAngleToCloserSide",
                "DistanceY", "Distance"]:
        if col in Data_drop.columns:
            del (Data_drop[col])
# else:
#     Data_drop["AngleMinusMin"] = Data_drop["AbsAngle"] - Data_drop["MinAngle"]
#     Data_drop["AngleMinusMax"] = Data_drop["AbsAngle"] - Data_drop["MaxAngle"]

for x in Data_drop.columns:
    print(x)

cols = list(Data_drop.columns.values)
cols.pop(cols.index('label'))
Features = cols
Data_drop = Data_drop[Data_drop["label"] != -1]
x_data = Data_drop.drop(['label'], axis=1)
y_data = Data_drop['label']
# correlations = x_data.corr()
# cor_matrix = correlations.abs()
# upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

# features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
# x_data = x_data.drop(columns=features_to_drop)
SEED = None
refcv_n_repeats = 1
# X = data.drop('label', axis=1)
X = x_data
target = y_data
# rfc = RandomForestClassifier(random_state=101)
# rfc = xgb.XGBClassifier(random_state=42)
rfc = xgb.XGBClassifier()
# rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv = RFECV(estimator=rfc, step=1,
              cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=refcv_n_repeats, random_state=SEED),
              scoring='accuracy')
rfecv.fit(X, target)
print('Optimal number of features: {}'.format(rfecv.n_features_))
print(np.where(rfecv.support_ == False)[0])

X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
dset = pd.DataFrame()
dset['attr'] = X.columns
dset['importance'] = rfecv.estimator_.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)
debug = False
if debug:
    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    plt.show()
best_features = [x for x in dset['attr']]
features_to_use = best_features[:50]
df_RFE = Data_drop[features_to_use + ['label']]
print(features_to_use)

X_train, X_test, y_train, y_test = train_test_split(Data_drop[features_to_use], Data_drop['label'], test_size=0.3,
                                                    random_state=23, stratify=Data_drop['label'])

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.7,
                                                              random_state=23, stratify=y_test)

# df_RFE = Data_drop[features_to_use[:15] + ['label']]
# X_train, X_test, y_train, y_test = train_test_split(Data_drop[features_to_use[:15]], Data_drop['label'], test_size=0.2,
#                                                     random_state=23, stratify=Data_drop['label'])
X_test_falses = Data_drop[features_to_use][Data_drop['label'] == 0]
scores = []
for i in range(20, 55, 5):
    for j in range(4, 11):
        model_RFE, best_score = model_training(X_train, X_validation, y_train, y_validation, i, j)
        precision = 1 - sum(model_RFE.predict(X_test_falses) / len(X_test_falses.index))
        scores.append([best_score["n_estimators"], best_score["depth"], best_score["fscore"], precision])

model_RFE, best_score = model_training(X_train, X_test, y_train, y_test, 50, 8)
accuracy_falses = 1 - np.sum(abs(model_RFE.predict(X_test_falses))) / len(X_test_falses.index)
y_pred = model_RFE.predict(X_test)
y_true = model_RFE.predict(X_test[y_test == 1])

accuracy_test_2 = 1 - np.sum(abs(y_test - y_pred)) / len(y_pred)
print("Falses accuracy ", accuracy_falses)


model_RFE.save_model(ModelFileName)
export_features(X_train)
# plt.rcParams['figure.figsize'] = [20, 10]
# plt.show()
# xgb.plot_importance(model_RFE)

df["Car"] = np.where(df['Class'] == 0, 1.0, 0.0)
df["Bus"] = np.where(df['Class'] == 1, 1.0, 0.0)
df["Truck"] = np.where(df['Class'] == 2, 1.0, 0.0)
df["MotorCycle"] = np.where(df['Class'] == 3, 1.0, 0.0)
df["Score"] = model_RFE.predict(df[X_train.columns])
df.to_csv("model_output.csv")
