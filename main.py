import os
import warnings
# import math
import matplotlib.pyplot as plt
# import numpy as np
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import gaussian_kde
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from FeatureExtractor import *

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


def export_features(feature_Vec):
    File = open('Features.txt', 'w')
    for feature in feature_Vec:
        if feature == 'label':
            continue
        if feature.find("Abs_") == -1 and not feature.lower() in \
                                              ["truck", "bus", "car", "MotorCycle", "motorcycle",
                                               "BoxCenterMovementInY", "DistanceToInnerSideDiff",
                                               "HeightOverWidthIntercept", "DistanceToInnerDiff",
                                               "BoxCenterMovementInX", "BoxCenterNormalizedMovementInX"]:
            feature = "Values." + feature
        print(feature + ",", file=File)
    File.close()


def plot_intersection_df(df, feature, quant=0.05):
    return plot_intersection(df[feature][df['label'] == 0], df[feature][df['label'] == 1], feature, quant)


def plot_intersection(x0, x1, title, quant=0.05):
    x0 = remove_outliers(x0, quant)
    x1 = remove_outliers(x1, quant)

    kde0 = gaussian_kde(x0, bw_method=0.3)
    kde1 = gaussian_kde(x1, bw_method=0.3)

    xmin = min(x0.min(), x1.min())
    xmax = max(x0.max(), x1.max())
    dx = 0.2 * (xmax - xmin)  # add a 20% margin, as the kde is wider than the data
    xmin -= dx
    xmax += dx

    x = np.linspace(xmin, xmax, 500)
    kde0_x = kde0(x)
    kde1_x = kde1(x)
    inters_x = np.minimum(kde0_x, kde1_x)

    plt.plot(x, kde0_x, color='b', label='False')
    plt.fill_between(x, kde0_x, 0, color='b', alpha=0.2)
    plt.plot(x, kde1_x, color='orange', label='True')
    plt.fill_between(x, kde1_x, 0, color='orange', alpha=0.2)
    plt.plot(x, inters_x, color='r')
    plt.fill_between(x, inters_x, 0, facecolor='none', edgecolor='r', hatch='xx', label='intersection')

    area_inters_x = np.trapz(inters_x, x)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels[2] += f': {area_inters_x * 100:.1f} %'
    plt.legend(handles, labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title + "_intersection.png")
    return area_inters_x


def calc_intersection(x0, x1):
    kde0 = gaussian_kde(x0, bw_method=0.3)
    kde1 = gaussian_kde(x1, bw_method=0.3)

    xmin = min(x0.min(), x1.min())
    xmax = max(x0.max(), x1.max())
    dx = 0.2 * (xmax - xmin)  # add a 20% margin, as the kde is wider than the data
    xmin -= dx
    xmax += dx

    x = np.linspace(xmin, xmax, 500)
    kde0_x = kde0(x)
    kde1_x = kde1(x)
    inters_x = np.minimum(kde0_x, kde1_x)
    area_inters_x = np.trapz(inters_x, x)
    return area_inters_x


def remove_outliers(vec, quant=0.05):
    vec[vec < np.quantile(vec, quant)] = np.quantile(vec, quant)
    vec[vec > np.quantile(vec, 1 - quant)] = np.quantile(vec, 1 - quant)
    return vec


def model_training(X_train, X_test, y_train, y_test, n_estimators=30, depth=5, Lambda=0):
    # random_state = 123
    accuracy_score_all = []
    precision_list_test = []
    recall_list_test = []
    fscore_list_test = []
    global model
    # del (X['label'])
    for i in range(0, 9):
        # print("Random_state", random_state)
        # Xt, Xv, yt, yv = train_test_split(X_SDA, y_SDA , test_size=0.3, random_state=random_state,stratify=y_SDA)

        # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
        # dt = xgb.DMatrix(Xt, label=yt.values)
        # dv = xgb.DMatrix(Xv, label=yv.values)
        # scale_pos_weight=-1 + 1 / np.mean(y),
        xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=depth,
                                      reg_lambda=Lambda)  # scale_pos_weight = 0.35,
        # default parameter for XGBoost classifier
        model = xgb_model.fit(X_train, y_train)

        pred_train = xgb_model.predict(X_train)
        pred_test = xgb_model.predict(X_test)
        # pred_val = xgb_model.predict(X_val)

        accuracy_train = accuracy_score(y_train, pred_train > 0.5)
        accuracy_test = accuracy_score(y_test, pred_test > 0.5)
        # accuracy_val = accuracy_score(y_val, pred_val>0.5)

        # print("Train Accuracy SCORE:", accuracy_train)
        # print("Test Accuracy SCORE:", accuracy_test)
        # print ("Val Accuracy SCORE:", accuracy_val)

        accuracy_score_all.append(accuracy_test)
        # accuracy_score_val.append(accuracy_val)

        precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(y_test, pred_test,
                                                                                      average='weighted')
        recall_list_test.append(recall_test)
        recall_list_test.append(fscore_test)

        # store the train performace values in list
        precision_list_test.append(precision_test)
        recall_list_test.append(recall_test)
        fscore_list_test.append(fscore_test)

        random_state = np.random.randint(100)
        # acc_xgb = (preds == yv).sum().astype(float) / len(preds)*100

    average_accuracy_score_all = np.mean(accuracy_score_all)
    average_precision_score_all = np.mean(precision_list_test)
    average_recall_score_all = np.mean(recall_list_test)
    average_fscore_score_all = np.mean(fscore_list_test)
    # average_accuracy_score_val= np.mean(accuracy_score_val)
    print("n_estimators = ", n_estimators, "max_depth = ", depth)
    print("Average test accuracy: ", average_accuracy_score_all)
    print("Average precison: ", average_precision_score_all)
    print("Average recall: ", average_recall_score_all)
    print("Average fscore: ", average_fscore_score_all)
    print()
    score = {"n_estimators": n_estimators, "depth": depth, "accuracy": average_accuracy_score_all,
             "precision": average_precision_score_all, "recall": average_recall_score_all,
             "fscore": average_fscore_score_all}
    return model, score


def check_accuracy(df, videos_directory, alert_string):
    video_files = [file for file in os.listdir(videos_directory) if file.lower().find(alert_string) != -1]
    df_passed = np.unique(df[df["ClassifierResultWithRules"] > 0.5]['Black Box Filename'])
    df_passed = [file for file in df_passed if file.lower().find(alert_string) != -1 and file.lower().find("false") == -1]
    missed_true = []
    for file in video_files:
        if file not in df_passed:
            missed_true.append(file)
    return missed_true


columns_to_drop = ['Black Box Frame Number', 'Black Box Filename', "Alert Type",
                   'TimeStamp', 'passed', 'Filename', 'alert_type', 'Frame_Number', 'Id', 'Class',
                   'ClassifierResult', 'Angle', "ClassifierResultWithRules", "AbsSpeedKMH", "TiltAngle",
                   "BoxLeft", "BoxRight", "MinLeft", "MinRight", "MaxLeft", "MaxRight", 'BoxLeftsFit.Intercept',
                   'BoxLeftsFit.Slope', 'BoxRightsFit.Intercept', 'BoxRightsFit.Slope', 'BoxWidth', 'BoxWidthPrev',
                   'BoxBottomPrev', 'BoxCenter', 'BoxCenterPrev', "DistanceToSideOuterPrev", "DistanceToSideInnerPrev"]

AlertType = 'Front_Collision' #'Front_Collision'  # 'BlindSpot' #'Side_Collision'  # 'Safe_Distance'
ModelFileName = AlertType.replace("_", "") + 'Model.json'
DataFileName = AlertType.replace("_", "") + ".csv"
TrainFileName = AlertType.replace("_", "") + "_Train.csv"
TestFileName = AlertType.replace("_", "") + "_Test.csv"
alert_string = "blind"
videos_directory = r"D:\Camera Roll\Alerts\Blindspot"

if AlertType == 'BlindSpots':
    AlertType = 'BlindSpot'  # 'Safe_Distance'
    videos_directory = r"D:\Camera Roll\Alerts\Blindspot"
    alert_string = "blind"
else:
    columns_to_drop = columns_to_drop + ["MinDistance", "MaxDistance", "MinBoxBottom", "MaxBoxBottom",
                                         "MinBoxArea", "MaxBoxArea", "MinAngle", "MaxAngle"]
    alert_string = "front"

    if AlertType == 'Safe_Distance':
        ModelFileName = 'SafeDistanceModel.json'
        alert_string = "safe"
    if AlertType == 'Side_Collision':
        AlertType = 'Front_Collision_Side'
        DataFileName = "FrontCollision.csv"

    videos_directory = r"D:\Camera Roll\Alerts\Front Alerts"

X = [20.079561,
20.322699,
20.514967,
20.650309,
20.696707,
20.668905,
20.582949,
20.454716,
20.293453,
20.212645,
20.136835,
20.011429,
19.84733,
19.601595,
19.280842,
19.037996,
18.272127,
17.688713,
17.877378,
17.910179,
17.831026,
17.846643,
17.89571,
17.389172,
17.381393,
15.540437,
14.687608,
13.831761,
13.021725,
12.373349,
11.925015,
11.64684,
11.543862,
11.541631,
11.534646,
11.524245,
11.466128,
11.331426,
11.321826,
11.412875,
11.495481,
11.506383,
11.532297,
11.605269,
11.575066,
11.537978,
11.495873,
11.441651,
11.380019,
11.317428,
11.234839,
11.037236,
10.892599,
10.732511,
10.588257,
10.506857,
10.430487,
10.355291,
10.286968,
10.202226,
10.153468,
10.113932,
10.063154,
10.059556,
10.084335,
10.079462,
10.035353,
10.011556,
10.092132,
10.127367,
10.094501,
10.033304,
9.958266,
9.834128,
9.684165,
9.575078,
9.487103,
9.377172,
9.382773,
9.516932,
9.545704,
9.573199,
9.618886,
9.703425,
9.793152,
9.816857,
9.64594,
9.698818,
9.782838,
9.592955,
8.556746,
8.460031,
8.335505,
8.130663,
7.957588,
7.795959,
7.67103,
7.590568,
7.428875,
7.337123,
7.189833,
7.022491,
6.874743,
6.684021,
6.495138,
6.355838,
6.261701,
6.167907,
6.157751,
6.143479,
6.097842,
6.012927,
5.911219,
5.791847,
5.65172,
5.503779,
5.376842,
]
Y = [0.651109590076903,
0.647039377066566,
0.642963738345186,
0.640324544514591,
0.637609692175568,
0.637547698022968,
0.63742033490777,
0.637039275643998,
0.634835744668809,
0.634738687251211,
0.634646546428799,
0.634725716702711,
0.634613394793399,
0.634707519652183,
0.639320323681369,
0.642636229481879,
0.642535307594908,
0.642708708138338,
0.643045545025614,
0.661657085243595,
0.691300759009253,
0.702400584534295,
0.702467700614332,
0.7035302247946,
0.703518746159146,
0.700175343822584,
0.699101703071594,
0.701308008988025,
0.74543936357413,
0.81380821126295,
0.8687248074619,
0.928343357897772,
0.968666069604213,
0.983873388992888,
0.989902731483706,
0.994041306108825,
0.996801341608455,
1.00101694147456,
1.00376434058758,
1.00375873214173,
1.00375362795725,
1.00375288120268,
0.992063210944234,
0.969942386830416,
0.95359904133946,
0.934428592438761,
0.911243824332248,
0.907639149560684,
0.907537670352586,
0.907432169875707,
0.90728978312164,
0.916726865880988,
0.924825116524304,
0.938706558328861,
0.953274285819969,
0.962144104891065,
0.965599188421674,
0.96865706922946,
0.972215893092227,
0.97286043647849,
0.972826136960265,
0.972797805072909,
0.972760732568182,
0.968600472854259,
0.963879583609146,
0.961502329291904,
0.95967633460567,
0.95627838060391,
0.953460747109719,
0.952331738791907,
0.94478457945766,
0.931628312988689,
0.925981604690432,
0.925703709539061,
0.925347284400411,
0.925072326211696,
0.931535661214893,
0.944867088398009,
0.956215684762134,
0.956422516261454,
0.956465336757346,
0.956505590847274,
0.949772154786698,
0.933074429811952,
0.924646393699461,
0.920053726082399,
0.912246942095794,
0.904831335298593,
0.901556076218354,
0.898584671901742,
0.920981815371641,
0.92016205399016,
0.919934991838085,
0.923627585547655,
0.92753523722093,
0.932029215538796,
0.936832536682138,
0.947457826763141,
0.97073068698633,
0.993029487180175,
1.02895067915508,
1.06755617771998,
1.07972848237009,
1.10695248896031,
1.16125087854789,
1.19963073547459,
1.22967990905431,
1.25763812457419,
1.26130442166515,
1.26259709156382,
1.26762258561848,
1.27522804580848,
1.28500677723758,
1.30746504484787,
1.35600998366274,
1.44498835354044,
1.56965779130369,
]
df = pd.read_csv(DataFileName)
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
if 'passed' in Data.columns:
    Data = Data[Data['passed'] == 1]

Data["Car"] = np.where(Data['Class'] == 0, 1.0, 0.0)
Data["Bus"] = np.where(Data['Class'] == 1, 1.0, 0.0)
Data["Truck"] = np.where(Data['Class'] == 2, 1.0, 0.0)
Data["MotorCycle"] = np.where(Data['Class'] == 3, 1.0, 0.0)

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

if AlertType != 'BlindSpot':
    for col in ['MaxBoxArea', 'MinBoxArea', 'MaxBoxBottom', 'MinBoxBottom', 'MaxAngle', 'MinAngle', 'MaxDistance',
                'MinDistance', 'Counter', 'MovementAngle', 'MovementRadius', 'MaxDistanceToSideOuter',
                'MaxDistanceToSideInner', "HeightOverWidthIntercept"]:
        if col in Data_drop.columns:
            del (Data_drop[col])
else:
    Data_drop["AngleMinusMin"] = Data_drop["AbsAngle"] - Data_drop["MinAngle"]
    Data_drop["AngleMinusMax"] = Data_drop["AbsAngle"] - Data_drop["MaxAngle"]

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

model_RFE, best_score = model_training(X_train, X_test, y_train, y_test, 40, 8)
accuracy_falses = 1 - np.sum(abs(model_RFE.predict(X_test_falses))) / len(X_test_falses.index)
y_pred = model_RFE.predict(X_test)

accuracy_test_2 = 1 - np.sum(abs(y_test - y_pred)) / len(y_pred)
print("Falses accuracy ", accuracy_falses)

print(df_RFE.columns)

model_RFE.save_model(ModelFileName)
export_features(X_train)
# plt.rcParams['figure.figsize'] = [20, 10]
# plt.show()
xgb.plot_importance(model_RFE)

df["Car"] = np.where(df['Class'] == 0, 1.0, 0.0)
df["Bus"] = np.where(df['Class'] == 1, 1.0, 0.0)
df["Truck"] = np.where(df['Class'] == 2, 1.0, 0.0)
df["MotorCycle"] = np.where(df['Class'] == 3, 1.0, 0.0)
df["Score"] = model_RFE.predict(df[X_train.columns])
df.to_csv("model_output.csv")
