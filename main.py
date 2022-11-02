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
