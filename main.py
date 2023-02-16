import warnings
from FeatureExtractor import *
from sklearn.feature_selection import RFECV
from DataTagger import *

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

columns_to_drop = ['NCars', "MinBoxArea", "MaxBoxArea", "MinDistance", "MaxDistance", 'Black Box Frame Number',
                   'Black Box Filename', "Alert Type",
                   'TimeStamp', 'Filename', 'alert_type', 'Frame_Number', 'Id', 'Class',
                   'ClassifierResult', 'Angle', "ClassifierResultWithRules", "AbsSpeedKMH", "TiltAngle",
                   "BoxLeft", "BoxRight", "MinLeft", "MinRight", "MaxLeft", "MaxRight", 'BoxLeftsFit.Intercept',
                   'BoxLeftsFit.Slope', 'BoxRightsFit.Intercept', 'BoxRightsFit.Slope', 'BoxWidth', 'BoxWidthPrev',
                   'BoxBottomPrev', 'BoxCenter', 'BoxCenterPrev', "DistanceToSideOuterPrev", "DistanceToSideInnerPrev"]

AlertType = 'BlindSpot'  # 'Front_Collision'  # 'BlindSpot' # 'Safe_Distance'
ModelFileName = AlertType.replace("_", "") + 'Model.json'
DataFileName = "FrontAlerts.csv"
# DataFileName = "BlindSpotsHeavyBike.csv"
reLabelData = False
optimize_featurs = False
OptimizeNFeatures = False
AddSpeed = False
if reLabelData:
    df = LabelData(DataFileName)
else:
    df = pd.read_csv(DataFileName)
if AddSpeed:
    speeds = pd.read_csv("Speeds.csv")
    speeds["Speed"] = speeds.apply(lambda x: correct_text(x["Speed"], x['text']), axis = 1)
    speeds.to_csv("Speeds.csv", index=False)
    df2 = pd.merge(df, speeds, on=['Black Box Filename', 'Black Box Frame Number'], how='left')
    df2.fillna(-1)
    df2.to_csv("FrontAlertsSpeeds.csv", index=False)
if AlertType.lower().find('blind') != - 1:
    videos_directory = r"D:\Camera Roll\Alerts\Blindspot"
    DataFileName = "BackAlerts.csv"
    TaggedDataFileName = "BackAlerts.csv"

    alert_string = "blind"
    df["MaxMinAngleCloserSide"] = df["MaxAngleToCloserSide"] - df["MinAngleToCloserSide"]

    df["AngleMinusMin"] = df["AbsAngle"] - df["MinAngle"]

    df["AngleToCloserSideMinusMin"] = abs(df["AngleToCloserSide"]) - abs(df["MinAngleToCloserSide"])
    df["MovementAngleTan"] = df["HorizontalMovement"] / (df["VerticalMovement"] + 0.000000000001)
    df["PredictedAngle"] = df["AbsAnglesFit.Intercept"] + 25 * df["AbsAnglesFit.Slope"]
    df["PredictedAngleToCloserSide"] = df["AbsAnglesToCloserSideFit.Intercept"] + \
                                       25 * df["AbsAnglesToCloserSideFit.Slope"]
    df["AnglesRatio"] = df["AbsAnglesFromHorizonFit.Slope"] / (df["AbsAnglesFit.Slope"] + 0.00000000001)
else:
    videos_directory = r"D:\Camera Roll\Alerts\Front Alerts"

    df["ZScore"] = (df["MomentarySpeed"] - df["RelativeSpeedKMH"]) / (df["VelocitySTD"] + 0.00000001)
    columns_to_drop = columns_to_drop + ["MinBoxBottom", "MaxBoxBottom", "MinAngle", "MaxAngle"]
    alert_string = "front"
    if AlertType == 'Safe_Distance':
        alert_string = "safe"
    DataFileName = "FrontAlerts.csv"

# df = add_features(df)
df = df.drop_duplicates()
df.replace('#DIV/0!', 0, inplace=True)
df.replace('#NAME?', 0, inplace=True)
df.replace('#REF!', 0, inplace=True)
df.replace('inf', 1000000, inplace=True)

# print(check_accuracy(df, videos_directory, alert_string))
if AlertType.lower().find('blind') == -1:
    Data = df.loc[df["Alert Type"] == AlertType]
else:
    Data = df.copy()

Data = Data[(Data['Label'] == 1) | (Data['Label'] == 0)]

Data = OneHotEncoding(Data)
Data_drop = RemoveUnwantedColumns(Data, columns_to_drop)

if AlertType != 'BlindSpots':
    for col in ['id_per_video', 'MaxBoxArea', 'MinBoxArea', 'MaxBoxBottom', 'MinBoxBottom', 'MaxAngle', 'MinAngle',
                'MaxDistance', 'MinDistance', 'Counter', 'MaxDistanceToSideOuter',
                'MaxDistanceToSideInner', "HeightOverWidthIntercept", "MaxAngleToCloserSide", "MinAngleToCloserSide"]:
        if col in Data_drop.columns:
            del (Data_drop[col])

for x in Data_drop.columns:
    print(x)

cols = list(Data_drop.columns.values)
cols.pop(cols.index('Label'))
Features = cols
Data_drop = Data_drop[(Data_drop["Label"] == 0) | (Data_drop["Label"] == 1)]

X = Data_drop.drop(['Label'], axis=1)
y_data = Data_drop['Label']
correlations = X.corr()
cor_matrix = correlations.abs()
upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

SEED = 11
refcv_n_repeats = 1
target = y_data
# rfc = RandomForestClassifier(random_state=101)
if optimize_featurs:
    rfc = xgb.XGBClassifier(random_state=SEED)
    rfecv = RFECV(estimator=rfc, step=1,
                  cv=RepeatedStratifiedKFold(n_splits=8, n_repeats=refcv_n_repeats, random_state=SEED),
                  scoring='accuracy')
    rfecv.fit(X, target)
    print('Optimal number of features: {}'.format(rfecv.n_features_))
    print(np.where(rfecv.support_ is False)[0])

    X.drop(X.columns[np.where(rfecv.support_ is False)[0]], axis=1, inplace=True)
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
    best_features = ['LaneSplitting', 'PredictedX', 'BoxInnerSideFit.Intercept', 'SideDetected', 'Car',
                     'MedianTimeToCollision', 'DistanceOverArea', 'Bus', 'AnglesSTD', 'BoxBottom', 'HorizontalMovement',
                     'RotationsFit.Slope', 'BoxTop', 'RotationsFit.Intercept', 'DistanceFromClosestCorner', 'DistanceX',
                     'DistanceY', 'MotorCycle', 'PhysicalRotation', 'Rotation', 'BoxOuterSideFit.Intercept',
                     'BoxCenterNormalizedMovementInX', 'BoxBottomFit.Intercept', 'MaxMinAngle8Frames', 'RearDetected',
                     'AbsMedianAngularVelocity', 'Distance', 'VerticalMovement', 'AngleFromTop',
                     'DistanceFromClosestCornerPower', 'RelativeSpeedKMH', 'VectorToSide', 'MedianDistance',
                     'AbsXFit.Slope', 'AngleToCloserSide', 'AngleFromHorizon', 'YFit.Intercept',
                     'AbsAnglesToFurtherSideFit.Intercept', 'MovementRadius', 'BoxAreasFit.Slope',
                     'AbsAnglesToCloserSideFit.Slope', 'BoxArea', 'Truck', 'BoxWidthsFit.Intercept',
                     'BoxBottomFit.Slope', 'AbsAnglesFit.Slope', 'AbsAngle']
    # best_features = ['LaneSplitting', 'PredictedX', 'BoxInnerSideFit.Intercept', 'SideDetected', 'Car', 'MedianTimeToCollision', 'DistanceOverArea', 'Bus', 'BoxBottom', 'HorizontalMovement', 'RotationsFit.Slope', 'BoxTop', 'DistanceFromClosestCorner', 'DistanceX', 'DistanceY', 'MotorCycle', 'PhysicalRotation', 'Rotation',  'MaxMinAngle8Frames', 'RearDetected', 'AbsMedianAngularVelocity', 'Distance', 'VerticalMovement', 'AngleFromTop', 'DistanceFromClosestCornerPower', 'RelativeSpeedKMH', 'VectorToSide', 'MedianDistance', 'AbsXFit.Slope', 'AngleToCloserSide', 'AngleFromHorizon', 'YFit.Intercept', 'AbsAnglesToFurtherSideFit.Intercept', 'MovementRadius',  'BoxAreasFit.Slope', 'AbsAnglesToCloserSideFit.Slope', 'BoxArea', 'Truck', 'BoxWidthsFit.Intercept',  'BoxBottomFit.Slope', 'AbsAnglesFit.Slope', 'AbsAngle']
# front alerts
# best_features = ['MinDistanceToSideOuter', 'InnerBoxDistanceToSide', 'MotorCycle', 'DistanceX', 'BoxArea', 'BoxHeightsFit.Intercept', 'AngleFromTop', 'BoxAreasFit.Intercept', 'MaxMinusAngleToCloserSide', 'Rotation', 'RotationsFit.Intercept', 'Car', 'MaxMinusBoxBottom', 'YFit.Intercept', 'DistanceOverArea', 'BoxInnerSideFit.Slope', 'DistanceY', 'SideDetected', 'BoxInnerSideFit.Intercept', 'AngleToCloserSide', 'BoxTop', 'BoxOuterSideFit.Intercept', 'PredictedX', 'BoxBottom', 'AbsAnglesFromTopFit.Slope', 'MaxMinusAngle', 'Distance', 'MedianDistance', 'AngleFromHorizon', 'AbsAnglesToFurtherSideFit.Intercept', 'XFitRadius', 'VectorToSide', 'AbsAnglesFromHorizonFit.Intercept', 'RelativeSpeedKMH', 'AbsAnglesFromTopFit.Intercept', 'AbsAnglesToCloserSideFit.Intercept', 'AngleMinusMin', 'MaxMinAngleToCloserSide', 'BoxBottomMinusMIn']

features_to_use = best_features

df_RFE = Data_drop[features_to_use + ['Label']]
print(features_to_use)

if OptimizeNFeatures:
    for i in range(0, 20):
        features = features_to_use[:-i]
        print(i)
        model_RFE = model_train(Data_drop, features, 120, 6)

OptimizeHyperPrams = False
if OptimizeHyperPrams:
    X_train, X_test, y_train, y_test = train_test_split(Data_drop[features_to_use], Data_drop['Label'], test_size=0.3,
                                                        random_state=23, stratify=Data_drop['Label'])

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

features = features_to_use
model_RFE = model_train(Data_drop, features, 40, 8)

model_RFE.save_model(ModelFileName)
export_features(Data_drop[features])

df = OneHotEncoding(df)
df["Score"] = model_RFE.predict(df[features])
df.to_csv("model_output.csv")

# plt.rcParams['figure.figsize'] = [20, 10]
# plt.show()
# xgb.plot_importance(model_RFE)
