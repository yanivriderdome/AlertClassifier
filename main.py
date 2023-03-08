import warnings
from FeatureExtractor import *
from sklearn.feature_selection import RFECV
from DataTagger import *
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

columns_to_drop = ['NCars', "MinBoxArea", "MaxBoxArea", "MinDistance", "MaxDistance", 'Black Box Frame Number',
                   'Black Box Filename', "Alert Type", "HeightVSWidthFactor",
                   'TimeStamp', 'Filename', 'alert_type', 'Frame_Number', 'Id',"AbsSpeedKmh",
                   'ClassifierResult', 'Angle', "ClassifierResultWithRules", "AbsSpeedKMH", "TiltAngle",
                   "BoxLeft", "BoxRight", 'BoxWidth', 'BoxWidthPrev', "ClostestVehicle",
                   'BoxBottomPrev', 'BoxCenter', 'BoxCenterPrev', "DistanceToSideOuterPrev", "DistanceToSideInnerPrev"]

AlertType = 'Front_Collision'  # 'Front_Collision'  # 'BlindSpot' # 'Safe_Distance'
ModelFileName = AlertType.replace("_", "") + 'Model.json'
DataFileName = "FrontAlerts.csv"
# DataFileName = "BlindSpotsHeavyBike.csv"
reLabelData = False
optimize_featurs = True
Optimize_NFeatures = False
AddSpeed = False

if reLabelData:
    df = LabelData(DataFileName)
else:
    df = pd.read_csv(DataFileName)

if AddSpeed:
    speeds = pd.read_csv("Tagged_Data/Speeds.csv")
    # speeds["Speed"] = speeds.apply(lambda x: correct_text(x["Speed"], x['text']), axis=1)
    # speeds.to_csv("Speeds.csv", index=False)
    df = pd.merge(df, speeds, on=['Black Box Filename', 'Black Box Frame Number'], how='left')
    df.fillna(-1,inplace=True)
    df.to_csv("FrontAlertsSpeeds.csv", index=False)

if AlertType.lower().find('blind') != - 1:
    videos_directory = r"D:\Camera Roll\Alerts\Blindspot"
    DataFileName = "BackAlerts.csv"
    TaggedDataFileName = "BackAlerts.csv"

    alert_string = "blind"
    df["MaxMinAngleCloserSide"] = df["MaxAngleToCloserSide"] - df["MinAngleToCloserSide"]
    df["AngleSlopeOverStd"] = df["AbsAnglesFit.Slope"]/df["AnglesSTD"]
    del(df["MedianTimeToCollision"])
    df["AngleMinusMin"] = df["AbsAngle"] - df["MinAngle"]
    df["AngleHorizonRatio"] = df["AngleFromHorizon"] / (df["AngleFromTop"] + 0.000000000000001)
    df["AnglesFit/BoxAreaFit"] = df["AbsAnglesFit.Slope"] / (df["BoxAreasFit.Slope"] + 0.000000000000001)
    df["BoxBottomMinusMax"] = df["BoxBottom"] - df["MaxBoxBottom"]

    df["AngleToCloserSideMinusMin"] = abs(df["AngleToCloserSide"]) - abs(df["MinAngleToCloserSide"])
    df["MovementAngleTan"] = df["HorizontalMovement"] / (df["VerticalMovement"] + 0.000000000001)
    df["PredictedAngle"] = df["AbsAnglesFit.Intercept"] + 25 * df["AbsAnglesFit.Slope"]
    df["PredictedAngleToCloserSide"] = df["AbsAnglesToCloserSideFit.Intercept"] + \
                                       25 * df["AbsAnglesToCloserSideFit.Slope"]
    df["AnglesRatio"] = df["AbsAnglesFromHorizonFit.Slope"] / (df["AbsAnglesFit.Slope"] + 0.00000000001)
else:
    videos_directory = r"D:\Camera Roll\Alerts\Front Alerts"

    df["ZScore"] = (df["MomentarySpeed"] - df["FinalSpeed"]) / (df["VelocitySTD"] + 0.00000001)
    columns_to_drop = columns_to_drop + ["MinBoxBottom", "MaxBoxBottom", "MinAngle", "MaxAngle"]
    alert_string = "front"
    if AlertType == 'Safe_Distance':
        alert_string = "safe"
    DataFileName = "FrontAlerts.csv"

df = df.drop_duplicates()
df.replace('#DIV/0!', 0, inplace=True)
df.replace('#NAME?', 0, inplace=True)
df.replace('#REF!', 0, inplace=True)
df.replace('inf', 1000000, inplace=True)
df = OneHotEncoding(df)
df["temp"] = df["Black Box Filename"].map(lambda x: x.find("Harley"))
df = df[df["temp"]==-1]
del(df["temp"])
if AlertType.lower().find('blind') == -1:
    Data = df.loc[df["Alert Type"] == AlertType]
else:
    Data = df.copy()

Data = Data[(Data['Label'] == 1) | (Data['Label'] == 0)]
Data_drop = RemoveUnwantedColumns(Data, columns_to_drop, AlertType)
[print(x) for x in Data_drop.columns]

cols = list(Data_drop.columns.values)
cols.pop(cols.index('Label'))
Features = cols
Data_drop.fillna(-1,inplace=True)
X = Data_drop.drop(['Label'], axis=1)
y = Data_drop['Label']

SEED = 11
refcv_n_repeats = 1
# rfc = RandomForestClassifier(random_state=101)
if optimize_featurs:
    rfc = xgb.XGBClassifier(random_state=SEED)
    rfecv = RFECV(estimator=rfc, step=1,
                  cv=RepeatedStratifiedKFold(n_splits=8, n_repeats=refcv_n_repeats, random_state=SEED),
                  scoring='accuracy')
    rfecv.fit(X, y)
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
        plt.xlabel('Importance', fontsize=14, Labelpad=20)
        plt.show()

    best_features = [x for x in dset['attr']]
else:
    # blind spots
    best_features = ['MinDistanceToSideOuter', 'BoxArea', 'BoxDistanceToSideInner', 'Class',
                     'BoxHeightsFit.Intercept', 'Rotation', 'DistanceOverArea', 'RotationsFit.Intercept',
                     'BoxHorizontalMovement', 'BoxAreasFit.Intercept', 'AngleFromTop', 'DistanceX', 'Orientation',
                     'DistanceToOuterSideDiff', 'BoxInnerSideFit.Slope', 'BoxWidthsFit.Slope',
                     'BoxInnerSideFit.Intercept', 'AngleToCloserSide', 'MedianDistance', 'DistancesFit.Intercept',
                     'MaxMinusAngleToCloserSide', 'DistanceY', 'VectorToSide', 'BoxWidthsFit.Intercept',
                     'AbsAnglesToCloserSideFit.Intercept', 'BoxTop', 'HorizontalMovement', 'MaxMinAngleCloserSide',
                     'BoxBottomFit.Intercept', 'BoxBottom', 'AbsAnglesFromHorizonFit.Intercept', 'AreaRatio8Frames',
                     'BoxOuterSideFit.Slope', 'RelativeSpeedKMH', 'AbsAnglesToFurtherSideFit.Slope',
                     'BoxAreasFit.Slope', 'BoxOuterSideFit.Intercept', 'MovementAngle', 'DistanceFromClosestCorner',
                     'YFit.Intercept', 'PhysicalRotation', 'IntersectionWithScreenIntercept',
                     'AbsMedianAngularVelocity', 'RotationsFit.Slope', 'XFitRadius']

    # best_features = ['LaneSplitting', 'PredictedX', 'BoxInnerSideFit.Intercept', 'SideDetected', 'Car', 'MedianTimeToCollision', 'DistanceOverArea', 'Bus', 'BoxBottom', 'HorizontalMovement', 'RotationsFit.Slope', 'BoxTop', 'DistanceFromClosestCorner', 'DistanceX', 'DistanceY', 'MotorCycle', 'PhysicalRotation', 'Rotation',  'MaxMinAngle8Frames', 'RearDetected', 'AbsMedianAngularVelocity', 'Distance', 'VerticalMovement', 'AngleFromTop', 'DistanceFromClosestCornerPower', 'RelativeSpeedKMH', 'VectorToSide', 'MedianDistance', 'AbsXFit.Slope', 'AngleToCloserSide', 'AngleFromHorizon', 'YFit.Intercept', 'AbsAnglesToFurtherSideFit.Intercept', 'MovementRadius',  'BoxAreasFit.Slope', 'AbsAnglesToCloserSideFit.Slope', 'BoxArea', 'Truck', 'BoxWidthsFit.Intercept',  'BoxBottomFit.Slope', 'AbsAnglesFit.Slope', 'AbsAngle']
# front 

features_to_use = best_features

df_RFE = Data_drop[features_to_use + ['Label']]
print(features_to_use)

OptimizeHyperPrams = False

features = features_to_use
# features = ['LaneSplitting', 'Car', 'AreaRatio8Frames', 'RearDetected', 'MaxDistanceX', 'VectorToSide', 'RotationsFit.Intercept', 'AngleFromTop', 'TimeToCollision', 'HorizontalMovement', 'FinalSpeed', 'AbsAngle', 'FrontDetected', 'BoxHeightsFit.Intercept', 'DistanceOverArea', 'SideDetected', 'AngleToCloserSide', 'BoxBottom', 'BoxWidthsFit.Slope', 'DistanceFromClosestCorner', 'MaxMinAngle8Frames', 'DistanceX', 'Bus', 'DistanceY', 'BoxTop', 'AbsAnglesToFurtherSideFit.Slope', 'BoxDistanceToSideInner', 'RotationsFit.Slope', 'AbsAnglesFromHorizonFit.Intercept', 'BoxBottomFit.Intercept', 'YFit.Slope', 'Rotation', 'XFitRadius', 'BoxInnerSideFit.Slope', 'AnglesSTD', 'MedianVelocity', 'BoxInnerSideFit.Intercept', 'BoxDistanceToSideOuter', 'AbsAnglesFit.Slope', 'MovementRadius', 'AbsAnglesToCloserSideFit.Intercept', 'BoxHorizontalMovement', 'AbsAnglesFromTopFit.Slope', 'VelocitySTD', 'BoxOuterSideFit.Intercept', 'BoxWidthsFit.Intercept', 'AbsXFit.Slope', 'AbsAnglesToFurtherSideFit.Intercept', 'BoxHeightsFit.Slope', 'BoxCenterMovementInX', 'MaxMinBoxArea8Frames', 'AbsAnglesFromTopFit.Intercept', 'DistanceFromClosestCornerPower', 'BoxAreasFit.Slope', 'AbsAnglesFit.Intercept', 'MaxMinDist8Frames', 'BoxCenterMovementInY', 'PredictedX', 'DistancesFit.Intercept', 'BoxAreasFit.Intercept', 'BoxVerticalMovement', 'VerticalMovement', 'AbsMedianAngularVelocity', 'Distance', 'AngleFromHorizon', 'MovementAngle', 'MedianTimeToCollision', 'PhysicalRotation', 'YFit.Intercept', 'BoxArea', 'AbsAnglesFromHorizonFit.Slope', 'Acceleration', 'DistanceXInCollision', 'BoxOuterSideFit.Slope', 'MedianDistance']
Data_drop["weights"] = Data["Black Box Filename"].map(lambda x: 4 if "Harley" in x else 1)

X_train, X_test, y_train, y_test = train_test_split(Data_drop[features_to_use + ['weights']], Data_drop['Label'], test_size=0.3,
                                                    random_state=23, stratify=Data_drop['Label'])
train_weights = X_train["weights"].copy()
del(X_train["weights"])
del(X_test["weights"])

dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
dtest = xgb.DMatrix(X_test, label=y_test)
xgb_params = {'max_depth': 8,  #
              'objective': 'binary:logistic',
              'eval_metric': 'aucpr'
              }

if OptimizeHyperPrams:
    num_round = 100
    xgbmodel = xgb.train(xgb_params, dtrain, num_round)

    y_pred = xgbmodel.predict(dtest)

    params_space = {'max_depth': hp.quniform("max_depth", 4, 11, 1),
                    'gamma': hp.uniform('gamma', 1, 9),
                    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                    'n_estimators': hp.quniform("n_estimators", 40, 120, 5),
                    'seed': 0
                    }

    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=params_space,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials)

    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)

param = {'max_depth': 8,  # Set the maximum depth of the trees to 8
         'objective': 'binary:logistic',  # Use binary logistic regression for binary classification
         'eval_metric': 'aucpr'  # Use AUCPR to evaluate the model's performance
         }
num_round = 100

xgbmodel = xgb.train(param, dtrain, num_round)

y_pred = xgbmodel.predict(dtest)
y_pred_binary = [1 if p >= 0.6 else 0 for p in y_pred]
n_falses = np.sum([y1 if y2 == 0 else 0 for y1, y2 in zip(y_pred_binary, y_test)])
n_missed = np.sum([1 - y1 if y2 == 1 else 0 for y1, y2 in zip(y_pred_binary, y_test)])

# Calculate the recall score
recall = recall_score(y_test, y_pred_binary)
print("recall = ", recall, "accuracy = ", accuracy_score(y_test, y_pred_binary), "n_missed = ", n_missed, "n_falses",
      n_falses)


xgbmodel.save_model(ModelFileName)
export_features(Data_drop[features_to_use])
all_data = xgb.DMatrix(df[features_to_use])
df["Score"] = xgbmodel.predict(all_data)
df.to_csv("model_output.csv", index=False)

# plt.rcParams['figure.figsize'] = [20, 10]
# plt.show()
# xgb.plot_importance(model_RFE)
