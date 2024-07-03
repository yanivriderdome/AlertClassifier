import warnings
from FeatureExtractor import *
from sklearn.feature_selection import RFECV
from DataTagger import *
from sklearn.metrics import recall_score
from scipy import stats
import random
from ray import tune
from catboost import CatBoostClassifier
import statistics
from xgboost import plot_tree
from datetime import datetime


date = datetime.today()
date_string = date.strftime("%Y-%m-%d")
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

ReLabelData = False
OptimizeFeatures = True
AddSpeed = False
OptimizeHyperPrams = False
TestCatBoost = False
columns_to_drop = ["EstimatedX", "MinBoxArea", "MaxBoxArea", "MinDistance", "MaxDistance", 'Black Box Frame Number',
                   'Black Box Filename', "Alert Type", "HeightVSWidthFactor", "Counter", "MedianVelocity",
                   "LabelPrevious", "LabelPrevious.1", "Label2", "NLane", "NLanes", "ClosestVehicle",
                   'ClassifierScoreWithRules', 'ClassifierScore', "GpsSpeedKmh", "MedianClassifierScore",
                   "RelativeAngularSpeed", "RelativeSpeedKMHFinalByLength", "RelativeSpeedKMHByWidth",
                   "RelativeSpeedKMHByHeight", 'label',"SafeDistanceThreshold","SafeDistanceTime", "PhysicalRotation",
                   'TimeStamp', 'Filename', 'alert_type', 'Frame_Number', 'Id', "TimeToCollision", "XFitRadius",
                   "BBoxDistanceToInnerSideDiff",'NLanesLabel', 'Nlanes','MinX', 'MaxX', "FinalSpeedKMHX"
                   'Classifier Score', 'Angle',  "TiltAngle", "Distance",
                   "MomentarySpeedY", "BBox.Right", "BBox.Left",'MeanDistanceX', "MinDistanceToSideOuter",
                   "BBoxLeft", "BBoxRight", 'BoxWidthPrev', "ClostestVehicle", "DistanceX", "DistanceY",
                   'BBoxDistanceToInnerSide', "Gps.SpeedKmh",
                   'BoxBottomPrev', 'BoxCenter', 'BoxCenterPrev', "DistanceToSideOuterPrev", "DistanceToSideInnerPrev",
                   "id_per_video", "NCarsLeft", "NCarsRight", "MomentarySpeed16", "MedianDistance","BBoxBottomVSCenterFit.Intercept"
                   "MedianDistanceX", "MedianDistanceY", "RelativeVelocitySTD", "RelativeSpeedKMHMean","Global.Gps.SpeedKmh",
                   "MaxBoxBottom", "Id_min", "IdPerVideo", "MidX", 'DistancesFit.Intercept']

AlertType = 'BlindSpots'  # 'Front_Collision' 'BlindSpots' 'Safe_Distance'

mode = 'Other' # 'Bikes', Other
ModelFileName = AlertType.replace("_", "") + 'Model.json'
DataFileName = AlertType.replace("_", "") + ".csv"
epsilon = 0.000000000000001
if ReLabelData:
    df = LabelData(DataFileName, AlertType)
    new_filename = DataFileName.replace(".csv", "_tagged_" + date_string + ".csv")
    df.to_csv(new_filename, index=False)
else:
    df = pd.read_csv(DataFileName)
for col in df.columns:
    if "TrafficStatistics" in col or "Safe" in col:
        del(df[col])
df["BBoxMovementRatio"] = df["MaxMinBBoxCenterXSeries"] / (df["MaxMinBBoxBottomSeries"] + 0.00000001)
if AlertType == 'BlindSpots':
    if mode == 'Bikes':
        df = df[df["Class"] == 3]

        ModelFileName = 'BlindSpotsModelBikes.json'
    else:
        df = df[df["Class"] != 3]
    # del(df["RelativeSpeedKMHY"])
faulty_columns = [x for x in df.columns if "Unnamed" in x] + ['id_per_video', 'Unnamed: 254', 'FirstAppearedFile',
                                                              'FirstAppearedFrameNumber']
df['BoxCenterXFit.Intercept'] = df['BoxCenterXFit.Intercept'].apply(lambda x: abs(x-0.5))
for x in faulty_columns:
    if x in df.columns:
        del (df[x])

original_columns = df.columns

if AddSpeed:
    speeds = pd.read_csv("Tagged_Data/Speeds.csv")
    df = pd.merge(df, speeds, on=['Black Box Filename', 'Black Box Frame Number'], how='left')
    df["Gps.SpeedKmh"] = df["Speed"]
    del (df["Speed"])
    del (df["text"])

    df.fillna(-1, inplace=True)
    new_filename = DataFileName.replace(".csv", "_tagged_NEW.csv")
    df.to_csv(new_filename, index=False)


def GetDistanceFromHorizonFit(BoxBottoms, BoxCenterX):
    BoxBottoms = [x - 0.4 for x in BoxBottoms if x != -1]
    BoxCenterX = [x - 0.5 for x in BoxCenterX if x != -1]
    distance = ([np.sqrt(xx * xx + yy * yy) for (xx, yy) in zip(BoxBottoms, BoxCenterX)])
    result = stats.theilslopes(distance)
    return result


def getAngleDerivative(df_angles):
    df_angles["Angles"] = df_angles.apply(
        lambda x: [x["Angles[" + str(i) + "]"] for i in range(16) if x["Angles[" + str(i) + "]"] != -1], axis=1)
    df_angles["AnglesFlatSlope"] = df_angles["Angles"].map(
        lambda lst: statistics.median([lst[i + 1] - lst[i] for i in range(len(lst) - 1)]))
    return df_angles


def filter_angles(row):
    return [angle for angle in row if angle != -1]


df.fillna(0, inplace=True)
if 'blind' in AlertType.lower():
    for column in ["Acceleration"]:
        del (df[column])
    df = df[df['ClassifierScoreWithRules'] >= -1]

    df["AngleSlopeOverStd"] = df["AbsAnglesFit.Slope"] / df["AnglesSTD"]
    df["AngleToCloserSideMinusMin"] = abs(df["AngleToCloserSide"]) - abs(df["MinAngleToCloserSide"])
    df["AnglesHorizonRatio"] = df["AbsAnglesFromHorizonFit.Slope"] / (df["AbsAnglesFit.Slope"] + epsilon)
    df["PredictedAngle"] = df["AbsAnglesFit.Intercept"] + 25 * df["AbsAnglesFit.Slope"]

    df["MaxMinAngleCloserSide"] = df["MaxAngleToCloserSide"] - df["MinAngleToCloserSide"]
    df["BoxAreaOverBoxBottom"] = df["BoxAreasFit.Slope"] / (df["BoxBottomFit.Slope"] + epsilon)

    del (df["MinAngle"])
    del (df["MaxAngle"])
    columns_to_drop = columns_to_drop + ["TimeToCollisionMedian","MomentarySpeed", "Global.Gps.SpeedKmh" , "InTheSameLaneByRear"]
else:

    # df["RearDetected"] = df.apply(lambda x: max(x["FrontDetected"],x["RearDetected"]), axis=1)
    # del(df["FrontDetected"])
    columns_to_drop = columns_to_drop + ["MaxMinusAngleToCloserSide", "MaxMinusMinAngle", "MaxAngleToCloserSide",
                                         "MaxMinusAngle", "MaxDistanceX", 'MaxBoxArea', 'MinBoxArea', 'MaxDistance',
                                         'MinDistance', 'Counter', 'MaxDistanceToSideOuter', 'MaxDistanceToSideInner',
                                         "HeightOverWidthIntercept",  "MinAngleToCloserSide",
                                         "MinBoxBottom", "MaxBoxBottom", "MinAngle", "MaxAngle", "AngleRange",
                                         "BBoxesHeightOverWidthFit.Slope", "SpeedCorrectionFactor"]

    if AlertType == "Safe_Distance":
        # df = df[df["InTheSameLaneByRear"] | ~df["RearDetected"] | df["AbsAngle"] < 8]
        columns_to_drop = columns_to_drop + ["BoxArea", "BoxBottom", "DistanceY", 'Distance', 'MedianDistance',
                                             'DistanceY', "DistanceOverArea",'YFit.Intercept', 'FinalDistance',
                                             'MedianDistanceY', 'FinalDistance', "BoxTop",'FinalDistanceY',
                                             'BoxAreasFit.Intercept', 'BoxBottomFit.Intercept','BoxWidthSide',
                                             'DistancesFromHorizonFit.Intercept', 'DistancesFit.Intercept',
                                             'BoxWidthFront', 'BoxAreasFit.Intercept','DistanceOverArea',
                                             'DistancesFromHorizonFit.Intercept', 'BoxArea', "DistanceFromClosestCornerPower"
                                             'BoxBottomFit.Intercept', 'BoxTop', 'BBoxWidthsFit.Intercept',
                                             "DistanceFromClosestCornerCorrected",
                                             'BoxWidthsFit.Intercept', 'BoxHeightsFit.Intercept', ]
df.fillna(0, inplace=True)
df.replace([float('inf'), -float('inf')], 0, inplace=True)

if "Unnamed: 248" in df.columns:
    del (df["Unnamed: 248"])
for column in ['Scores[1]', 'Scores[2]', 'Scores[3]', 'Scores[4]', 'Scores[5]',
               'Scores[6]', 'Scores[7]', 'Id_min', 'IdPerVideo', "BBoxDistanceToOuterSideMean.1",
               "BBoxVerticalMovement",
               "BBoxHorizontalMovement", "BBoxDistanceToOuterSide"]:
    if column in df.columns:
        del df[column]
for column in ['TimeDeltas[1]', 'TimeDeltas[2]', 'TimeDeltas[3]', 'TimeDeltas[4]', 'TimeDeltas[5]',
               'TimeDeltas[6]', 'TimeDeltas[7]', 'TimeDeltas[8]', 'TimeDeltas[9]', 'TimeDeltas[10]', 'TimeDeltas[11]',
               'TimeDeltas[12]',
               'TimeDeltas[13]', 'TimeDeltas[14]', 'TimeDeltas[15]']:
    if column in df.columns:
        del df[column]

df = df.drop_duplicates()
df.replace('#DIV/0!', 0, inplace=True)
df.replace('#NAME?', 0, inplace=True)
df.replace('#REF!', 0, inplace=True)
df.replace('inf', 1000000, inplace=True)
df = OneHotEncoding(df)

# if AlertType.lower().find('blind') == -1:
#     Data = df.loc[df["Alert Type"] == AlertType]
# else:
#     Data = df.copy()

df = df[(df['Label'] == 1) | (df['Label'] == 0)]
df['Label'] = df['Label'].apply(lambda x: float(x))

Data_drop = RemoveUnwantedColumns(df, columns_to_drop, AlertType)

cols = list(Data_drop.columns.values)
cols.pop(cols.index('Label'))
Features = cols
Data_drop.fillna(-1, inplace=True)

X = Data_drop.drop(['Label'], axis=1)
y = [float(yy) for yy in Data_drop['Label']]

[print(x) for x in X.columns]

SEED = 11
refcv_n_repeats = 1
# rfc = RandomForestClassifier(random_state=101)
if OptimizeFeatures:

    rfc = xgb.XGBClassifier(random_state=SEED)
    rfecv = RFECV(estimator=rfc, step=1,
                  cv=RepeatedStratifiedKFold(n_splits=8, n_repeats=refcv_n_repeats, random_state=SEED),
                  scoring='accuracy')
    X.dropna(inplace=True)
    X = X.apply(lambda col: col.apply(lambda x: x if x != '[]' and x!= "`" else 0))

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
    best_features = get_features_from_file(ModelFileName)

df_RFE = Data_drop[best_features + ['Label']]
print(best_features)
# df_RFE["weights"] = df_RFE["Black Box Filename"].map(lambda x: 1 if "Harley" in x else 1)
df_RFE.fillna(0)
filenames = list(df["Black Box Filename"].unique())
Train_set_size = int(len(filenames) * 0.8)
train_files = random.sample(filenames, Train_set_size)
test_files = [x for x in filenames if x not in train_files]

train = df_RFE[df["Black Box Filename"].isin(train_files)]
test = df_RFE[df["Black Box Filename"].isin(test_files)]
X_train = train[best_features]  # + ['weights']
X_test = test[best_features]
y_train = train['Label']
y_test = test['Label']

# X_train, X_test, y_train, y_test = train_test_split(Data_drop[best_features + ['weights']], Data_drop['Label'],
#                                                     test_size=0.3,
#                                                     random_state=23, stratify=Data_drop['Label'])
# train_weights = X_train["weights"].copy()

if OptimizeHyperPrams:
    def optimize_xgb(configuration):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': configuration['max_depth'],
            'subsample': configuration['subsample'],
            'eta': configuration['learning_rate'],
            'gamma': configuration['gamma']
        }

        # Create the XGBoost classifier
        xgb_clf = xgb.XGBClassifier(**params)

        # Train the model
        xgb_clf.fit(configuration["X"], configuration["y"])

        # Compute the validation accuracy
        accuracy = xgb_clf.score(configuration["X"], configuration["y"])

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

    params = {'max_depth': analysis.best_config['max_depth'],
              'gamma': analysis.best_config['gamma'],
              'learning_rate': analysis.best_config['learning_rate'],
              'subsample': analysis.best_config['subsample'],
              }
else:
    if AlertType == 'BlindSpots':
        params = {'max_depth': 8}
    else:
        params = {'max_depth': 7}


if os.path.exists(ModelFileName):
    with open(ModelFileName, 'r') as json_file:
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(ModelFileName)


xgbmodel = model_train(Data_drop, best_features, AlertType, params, true_weight = 1)

xgbmodel.save_model(ModelFileName)
all_data = xgb.DMatrix(Data_drop[best_features])
df["OldClassifier Score"] = df['ClassifierScore']
df['ClassifierScore'] = xgbmodel.predict(all_data)
df.to_csv("model_output.csv", index=False)

def generate_lines(filename, features):
    with open(filename, 'w') as file:
        for item in features:
            line = f'Model->SetFeature("{item}", Values.{item});\n'
            file.write(line)

# Draw trees
DrawTrees = False
if DrawTrees:
    plot_tree(xgbmodel, num_trees=0, rankdir='LR')
    df["Class"] = Data_drop["Class"]

# CompareToPrev = False

# if CompareToPrev:
#     prev_model_filename = "FrontCollisionModel_old.json"
#     if os.path.exists(prev_model_filename):
#         df["Classifier ScorePrevC"] = PredictFromFile(prev_model_filename, df)
#
#     prev_model_filename = "FrontCollisionModel.json"
#     df["Classifier Score"] = PredictFromFile(ModelFileName, df)
#
#     print("threshold = ", calc_threshold(df, AlertType))

if TestCatBoost:
    clf = CatBoostClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_pred_binary = [1 if p >= 0.6 else 0 for p in y_pred]
    n_falses = np.sum([y1 if y2 == 0 else 0 for y1, y2 in zip(y_pred_binary, y_test)])
    n_missed = np.sum([1 - y1 if y2 == 1 else 0 for y1, y2 in zip(y_pred_binary, y_test)])

    # Calculate the recall score
    recall = recall_score(y_test, y_pred_binary)
    print("recall = ", recall, "accuracy = ", accuracy_score(y_test, y_pred_binary), "n_missed = ", n_missed,
          "n_falses",
          n_falses)
    print('Accuracy:', clf.score(X_test, y_test))

    # plt.rcParams['figure.figsize'] = [20, 10]
    # plt.show()
    # xgb.plot_importance(model_RFE)
