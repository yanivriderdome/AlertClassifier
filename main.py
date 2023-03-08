import warnings
from FeatureExtractor import *
from sklearn.feature_selection import RFECV
from DataTagger import *
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import random
from ray import tune

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

columns_to_drop = ['NCars', "MinBoxArea", "MaxBoxArea", "MinDistance", "MaxDistance", 'Black Box Frame Number',
                   'Black Box Filename', "Alert Type", "HeightVSWidthFactor",
                   'TimeStamp', 'Filename', 'alert_type', 'Frame_Number', 'Id', "AbsSpeedKmh",
                   'ClassifierResult', 'Angle', "ClassifierResultWithRules", "AbsSpeedKMH", "TiltAngle",
                   "BoxLeft", "BoxRight", 'BoxWidth', 'BoxWidthPrev', "ClostestVehicle",
                   'BoxBottomPrev', 'BoxCenter', 'BoxCenterPrev', "DistanceToSideOuterPrev", "DistanceToSideInnerPrev"]

AlertType = 'BlindSpots'  # 'Front_Collision'  # 'BlindSpot' # 'Safe_Distance'
ModelFileName = AlertType.replace("_", "") + 'Model.json'
DataFileName = "BlindSpots.csv"
# DataFileName = "BlindSpotsHeavyBike.csv"
reLabelData = False
optimize_featurs = False
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
    df.fillna(-1, inplace=True)
    df.to_csv("FrontAlertsSpeeds.csv", index=False)


def GetDistanceFromHorizonFit(BoxBottoms, BoxCenterX):
    BoxBottoms = [x - 0.4 for x in BoxBottoms if x != -1]
    BoxCenterX = [x - 0.5 for x in BoxCenterX if x != -1]
    distance = ([np.sqrt(x * x + y * y) for (x, y) in zip(BoxBottoms, BoxCenterX)])
    result = stats.theilslopes(distance)
    return result


if AlertType.lower().find('blind') != - 1:
    for column in ["MedianVelocity", "TimeToCollision", "TimeToCollisionFullSeries", "Acceleration",
                   "MomentarySpeed16"]:
        del (df[column])
    df["DistancesFromHorizonFit.Slope"] = df.apply(
        lambda x: GetDistanceFromHorizonFit([x["BoxBottoms[" + str(i) + "]"] for i in range(8)],
                                            [x["BoxCenters[" + str(i) + "]"] / 100 for i in range(8)])[0], axis=1)
    df["DistancesFromHorizonFit.Intercept"] = df.apply(
        lambda x: GetDistanceFromHorizonFit([x["BoxBottoms[" + str(i) + "]"] for i in range(8)],
                                            [x["BoxCenters[" + str(i) + "]"] / 100 for i in range(8)])[1], axis=1)

    df["MaxMinAngleCloserSide"] = df["MaxAngleToCloserSide"] - df["MinAngleToCloserSide"]
    df["AngleSlopeOverStd"] = df["AbsAnglesFit.Slope"] / df["AnglesSTD"]
    df["DistanceToVehicleSide"] = df.apply(lambda x: x["Distance"] * np.cos(abs(x["AngleToCloserSide"])), axis=1)
    df["AngleMinusMin"] = df["AbsAngle"] - df["MinAngle"]
    df["AngleHorizonRatio"] = df["AngleFromHorizon"] / (df["AngleFromTop"] + 0.000000000000001)
    df["AnglesFit/BoxAreaFit"] = df["AbsAnglesFit.Slope"] / (df["BoxAreasFit.Slope"] + 0.000000000000001)
    df["BoxBottomMinusMax"] = df["BoxBottom"] - df["MaxBoxBottom"]
    df["BoxBottomMinusMin"] = df["BoxBottom"] - df["MinBoxBottom"]
    df["BoxDistanceToSideMinusMin"] = df["BoxDistanceToSideOuter"] - df["MinDistanceToSideOuter"]

    df["AngleToCloserSideMinusMin"] = abs(df["AngleToCloserSide"]) - abs(df["MinAngleToCloserSide"])
    df["MovementAngleTan"] = df["HorizontalMovement"] / (df["VerticalMovement"] + 0.000000000001)
    df["PredictedAngle"] = df["AbsAnglesFit.Intercept"] + 25 * df["AbsAnglesFit.Slope"]
    df["PredictedAngleToCloserSide"] = df["AbsAnglesToCloserSideFit.Intercept"] + \
                                       25 * df["AbsAnglesToCloserSideFit.Slope"]
    df["AnglesRatio"] = df["AbsAnglesFromHorizonFit.Slope"] / (df["AbsAnglesFit.Slope"] + 0.00000000001)
else:
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
df = df[df["temp"] == -1]
del (df["temp"])
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
Data_drop.fillna(-1, inplace=True)
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

# front
best_features = ['MinDistanceToSideOuter', 'MaxDistanceToSideOuter', 'BoxDistanceToSideInner', 'MotorCycle', 'DistanceOverArea', 'VectorToSide', 'BoxWidthFront', 'RotationsFit.Intercept', 'BoxWidthsFit.Intercept', 'BoxHorizontalMovement', 'Truck', 'BoxWidthSide', 'Rotation', 'BoxBottomMinusMax', 'AngleHorizonRatio', 'BoxWidthsFit.Slope', 'MaxAngleToCloserSide', 'BoxHeightsFit.Slope', 'MaxMinusAngle', 'MaxMinusAngleToCloserSide', 'TimeToCollisionBoxAreaFullSeries', 'AngleFromTop', 'Orientation', 'Distance', 'BoxBottomFit.Intercept', 'AbsAnglesFromHorizonFit.Intercept', 'DistanceX', 'MinAngle', 'MinBoxBottom', 'BoxOuterSideFit.Intercept', 'XFitRadius', 'RotationsFit.Slope', 'MaxDistanceX', 'AbsAnglesToFurtherSideFit.Slope', 'MaxBoxBottom', 'BoxBottom', 'MedianDistance', 'MaxMinAngleCloserSide', 'DistanceFromClosestCornerPower', 'MinAngleToCloserSide', 'MaxAngle', 'BoxAreasFit.Slope', 'MaxMinDist8Frames', 'HorizontalMovement', 'IntersectionWithScreenIntercept', 'BoxArea', 'DistanceToOuterSideDiff', 'AbsAnglesFromTopFit.Intercept', 'BoxInnerSideFit.Slope', 'BoxTop', 'BoxHeightsFit.Intercept', 'BoxOuterSideFit.Slope', 'BoxAreasFit.Intercept', 'AbsAnglesFromTopFit.Slope', 'PredictedX', 'BoxBottomMinusMin', 'DistanceY', 'BoxInnerSideFit.Intercept', 'AreaRatio8Frames', 'FinalSpeed', 'AngleFromHorizon', 'DistancesFit.Intercept', 'BoxCenterNormalizedMovementInX', 'AngleMinusMin', 'YFit.Slope', 'Car']

features_to_use = best_features

df_RFE = Data_drop[features_to_use + ['Label']]
print(features_to_use)

OptimizeHyperPrams = True

features = features_to_use

Data_drop["weights"] = Data["Black Box Filename"].map(lambda x: 1 if "Harley" in x else 1)

Data_drop["Black Box Filename"] = Data["Black Box Filename"]
filenames = list(Data["Black Box Filename"].unique())
Train_set_size = int(len(filenames) * 0.8)
train_files = random.sample(filenames, Train_set_size)
test_files = [x for x in filenames if x not in train_files]

train = Data_drop[Data["Black Box Filename"].isin(train_files)]
test = Data_drop[Data["Black Box Filename"].isin(test_files)]
X_train = train[features_to_use + ['weights']]
X_test = test[features_to_use]
y_train = train['Label']
y_test = test['Label']

# X_train, X_test, y_train, y_test = train_test_split(Data_drop[features_to_use + ['weights']], Data_drop['Label'],
#                                                     test_size=0.3,
#                                                     random_state=23, stratify=Data_drop['Label'])
# train_weights = X_train["weights"].copy()

if OptimizeHyperPrams:
    def optimize_xgb(config, checkpoint_dir=None, data=None):
        # Load the breast cancer dataset
        # Set the XGBoost parameters from the config
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': config['max_depth'],
            'subsample': config['subsample'],
            'eta': config['learning_rate'],
            'gamma': config['gamma']
        }

        # Create the XGBoost classifier
        xgb_clf = xgb.XGBClassifier(**params)

        # Train the model
        xgb_clf.fit(config["X"], config["y"])

        # Compute the validation accuracy
        accuracy = xgb_clf.score(config["X"], config["y"])

        # Save the checkpoint for future resumption
        with tune.checkpoint_dir(step=0) as checkpoint_dir:
            path = f"{checkpoint_dir}/checkpoint"
            xgb_clf.save_model(path)

        tune.report(accuracy=accuracy)


    config = {'max_depth': tune.randint(4, 11),
              'subsample': tune.uniform(0.5, 1.0),
              'learning_rate': tune.loguniform(1e-4, 1e-1),
              'gamma': tune.uniform(0.0, 1.0),
              "X": X_train,
              "y": y_train}
    # Configure the Ray Tune search
    analysis = tune.run(
        optimize_xgb,
        config=config,
        num_samples=10,
        resources_per_trial={'cpu': 1},
        metric='accuracy',
        mode='max',
        local_dir='./ray_results'
    )

    param = {'max_depth': analysis.best_config['max_depth'],
             'gamma': 0.448706,
             'learning_rate': 0.0142558,
             'subsample': 0.942048,
             'objective': 'binary:logistic',  # Use binary logistic regression for binary classification
             'eval_metric': 'aucpr'  # Use AUCPR to evaluate the model's performance
             }
else:

    param = {'max_depth': 8,  # Set the maximum depth of the trees to 8
             'objective': 'binary:logistic',  # Use binary logistic regression for binary classification
             'eval_metric': 'aucpr'  # Use AUCPR to evaluate the model's performance
             }
num_round = 100
if "weights" in X_train.columns:
    del (X_train["weights"])

train_weights = [1 if x == 0 else 1.2 for x in y_train]
dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)  #
dtest = xgb.DMatrix(X_test, label=y_test)

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
