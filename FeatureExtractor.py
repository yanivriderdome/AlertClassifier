import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import math
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import csv
from datetime import datetime
import json
import pandas as pd

def get_speed_from_text(text):
    parsed_text = str(text).split(" ")
    if len(parsed_text) > 2:
        found = False
        for i, x in enumerate(parsed_text):
            if x.find('km') != -1 and i > 0:
                speed = parsed_text[i - 1]
                speed = "".join([char for char in speed if char.isdigit() or char == "."])
                found = True
                break
        if not found:
            try:
                speed = float(speed)
            except:
                return -1
        return speed
    else:
        found = False
        for x in parsed_text:
            if x.find('km') != -1:
                speed = x.split('km')[0]
                found = True
                break
        if not found:
            for x in parsed_text[::-1]:
                try:
                    return float(x)
                except:
                    if x.find(":") == -1:
                        new_string = "".join([new_char for new_char in x if (new_char.isdigit() or new_char == '.')])
                        if len(new_string) > 0:
                            try:
                                return float(new_string)
                            except:
                                pass
            speed = parsed_text[0]

    return speed


def correct_text(speed, text):
    if text is None and speed is None:
        return -1
    try:
        float(speed)
        if speed > 100:
            return speed / 10
        return speed
    except:
        return get_speed_from_text(text)


def CalculateCounter(x1, x2, x3):
    if x3 == 0:
        return 5
    if x2 == 0:
        return 6
    if x1 == 0:
        return 7
    return 8


def NotBeingOverTaken(MaxDistanceToSide, MovementBoxInner, MovementBoxRight):
    if (MaxDistanceToSide < 5 or MovementBoxRight < 0) and MovementBoxInner < 0:
        return 0
    return 1


def CalculateHorizontalMovement(BoxLeft, BoxRight, BoxLeftPrev, BoxRightPrev):
    if BoxLeft < 5:
        return (BoxRight - BoxRightPrev) / 2
    if BoxRight > 95:
        return (BoxLeft - BoxLeftPrev) / 2

    return (BoxRight - BoxRightPrev + BoxLeft - BoxLeftPrev) / 4


def get_vec(x0, x1, x2, x3, x4, x5, x6, x7, counter, angle, shift=0):
    vec = [x0, x1, x2, x3, x4, x5, x6, x7]
    vec = [(x - shift) * np.sign(angle) for x in vec]
    return vec[:counter]


def get_curvature(X, Y):
    dY = np.asarray(Y[1:]) - np.asarray(Y[:-1])
    dX = np.asarray(X[1:]) - np.asarray(X[:-1])
    #
    # if np.mean(X) < 50.0:
    #     dX = np.asarray(X[1:]) - np.asarray(X[:-1])
    # else:
    #     dX = np.asarray(X[:-1]) - np.asarray(X[1:])
    ddX = np.asarray(dX[:-1]) - np.asarray(dX[1:])
    ddY = np.asarray(dY[:-1]) - np.asarray(dY[1:])
    k = 0
    for i in range(len(ddX)):
        denominator = dX[i + 1] ** 2 + dY[i + 1] ** 2
        k += abs(dX[i + 1] * ddY[i] - ddX[i] * dY[i + 1]) / pow(denominator, 3 / 2)
    return k


def get_Xcoordinate(H1, H2, H3, H4, H5, H6, H7, H8, factor):
    return np.asarray([h * factor for h in [H1, H2, H3, H4, H5, H6, H7, H8]])


def OneHotEncoding(Data_orig):
    Data = Data_orig.copy()
    # Data = Data[Data["Counter"] == 8]
    Data["Car"] = np.where(Data['Class'] == 0, 1.0, 0.0)
    Data["Bus"] = np.where(Data['Class'] == 1, 1.0, 0.0)
    Data["Truck"] = np.where(Data['Class'] == 2, 1.0, 0.0)
    Data["Motorcycle"] = np.where(Data['Class'] == 3, 1.0, 0.0)
    del (Data["Class"])
    return Data


def ConvertUnits(df, ConversionFactor=1.5):
    keys = [
        "Distance",
        "DistanceX",
        "DistanceY",
        "MedianDistance",
        "MedianDistanceX",
        "MedianDistanceY",
        "FinalDistance",
        "FinalDistanceX",
        "FinalDistanceY",
        "DistanceXToFurtherSide",
        "MomentarySpeed",
        "MomentarySpeed16",
        "MedianVelocity",
        "RelativeSpeedKMH",
        "AbsXFit.Slope",
        "AbsXFit.Intercept",
        "YFit.Slope",
        "YFit.Intercept",
        "DistancesFit.Slope",
        "DistancesFit.Intercept",
        "PredictedX",
        "PredictedFinalX",
        "MaxMinDist8Frames",
        "MaxMinDistXSeries",
        "DistanceOverArea",
        "XFitRadius"
    ]
    for column in keys:
        if column in df.columns:
            df[column] = df[column] * ConversionFactor
    return df


def ConvertUnitsButX(df, ConversionFactor=1.5):
    keys = [
        "Distance",
        "DistanceY",
        "MedianDistance",
        "MedianDistanceY",
        "FinalDistance",
        "FinalDistanceY",
        "DistanceXToFurtherSide",
        "MomentarySpeed",
        "MomentarySpeed16",
        "MedianVelocity",
        "RelativeSpeedKMH",
        "YFit.Slope",
        "YFit.Intercept",
        "DistancesFit.Slope",
        "DistancesFit.Intercept",
        "PredictedFinalX",
        "DistanceOverArea",
        "XFitRadius"
    ]
    for column in keys:
        if column in df.columns:
            df[column] = df[column] * ConversionFactor
    return df


def RemoveUnwantedColumns(Data, columns_to_drop, AlertType):
    Data_drop = Data.copy()
    for col in Data_drop.columns:
        if "score" in col.lower():
            del (Data_drop[col])

    for col in columns_to_drop:
        if col in Data_drop.columns:
            del (Data_drop[col])
    for i in range(8):
        if 'Scores[' + str(i) + ']' in Data_drop.columns or "LabelPrevious" in Data_drop.columns:
            del (Data_drop['Scores[' + str(i) + ']'])

    for i in range(16):
        if 'Distances[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['Distances[' + str(i) + ']'])

        if 'BoxWidths[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['BoxWidths[' + str(i) + ']'])
        if 'BoxHeights[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['BoxHeights[' + str(i) + ']'])
        if 'BoxCenters[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['BoxCenters[' + str(i) + ']'])
        if 'BoxBottoms[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['BoxBottoms[' + str(i) + ']'])
        if 'Angles[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['Angles[' + str(i) + ']'])
        if 'Angles[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['Angles[' + str(i) + ']'])
        if 'TimeDeltas[' + str(i) + ']' in Data_drop.columns:
            del (Data_drop['TimeDeltas[' + str(i) + ']'])
    for col in Data_drop.columns:
        if "[" in col and "]" in col:
            del (Data_drop[col])
    return Data_drop


def get_curvatureX(X, Y):
    dY = np.asarray(Y[1:]) - np.asarray(Y[:-1])
    dX = np.asarray(X[1:]) - np.asarray(X[:-1])
    #
    # if np.mean(X) < 50.0:
    #     dX = np.asarray(X[1:]) - np.asarray(X[:-1])
    # else:
    #     dX = np.asarray(X[:-1]) - np.asarray(X[1:])
    ddX = np.asarray(dX[:-1]) - np.asarray(dX[1:])
    ddY = np.asarray(dY[:-1]) - np.asarray(dY[1:])
    k = 0
    for i in range(len(ddX)):
        denominator = dX[i + 1] ** 2 + dY[i + 1] ** 2
        k += abs(dX[i + 1] * ddY[i] - ddX[i] * dY[i + 1]) / pow(denominator, 3 / 2)
    return k


def add_features(df):
    df["MaxDistanceToSideOuter"] = df.apply(lambda x: min(x["MaxLeft"], 100 - x["MinRight"]), axis=1)
    df["DistanceToSideOuter"] = df.apply(lambda x: min(x["BoxLeft"], 100 - x["BoxRight"]), axis=1)
    df['Counter'] = df.apply(lambda x: CalculateCounter(x["TimeDeltas[7]"], x["TimeDeltas[6]"], x["TimeDeltas[5]"]),
                             axis=1)
    df = df[df['Counter'] == 8]

    df["temp"] = "BoxWidths[" + (df['Counter'] - 1).apply(str) + "]"
    df['BoxWidth'] = df.apply(lambda x: x[x['temp']], 1)
    df["temp"] = "BoxWidths[" + (df['Counter'] - 2).apply(str) + "]"
    df['BoxWidthPrev'] = df.apply(lambda x: x[x['temp']], 1)

    df["temp"] = "BoxBottoms[" + (df['Counter'] - 1).apply(str) + "]"
    df['BoxBottom'] = df.apply(lambda x: x[x['temp']], 1)
    df["temp"] = "BoxBottoms[" + (df['Counter'] - 2).apply(str) + "]"
    df['BoxBottomPrev'] = df.apply(lambda x: x[x['temp']], 1)

    df["temp"] = "BoxCenters[" + (df['Counter'] - 1).apply(str) + "]"
    df['BoxCenter'] = df.apply(lambda x: x[x['temp']], 1)
    df["temp"] = "BoxCenters[" + (df['Counter'] - 2).apply(str) + "]"
    df['BoxCenterPrev'] = df.apply(lambda x: x[x['temp']], 1)
    df['BoxLeftPrev'] = df['BoxCenterPrev'] - df['BoxWidthPrev'] / 2
    df['BoxRightPrev'] = df['BoxCenterPrev'] + df['BoxWidthPrev'] / 2
    df["DistanceToSideOuterPrev"] = df.apply(lambda x: min(x["BoxLeftPrev"], 100 - x["BoxRightPrev"]), axis=1)
    df["DistanceToSideInnerPrev"] = df["DistanceToSideOuterPrev"] + df['BoxWidthPrev']

    df["DistanceToSideInnerDiff"] = np.sign(df["Angle"]) * (df["DistanceToSideInner"] - df["DistanceToSideInnerPrev"])
    df["DistanceToSideOuterDiff"] = np.sign(df["Angle"]) * (
            df["DistanceToSideInnerDiff"] + df['BoxWidthPrev'] - df['BoxWidth'])

    df["NotBeingOverTaken"] = df.apply(
        lambda x: NotBeingOverTaken(x["MaxDistanceToSideOuter"], x["DistanceToSideInnerDiff"],
                                    x["DistanceToSideOuterDiff"]), axis=1)

    df["BoxLeft0"] = (df["BoxCenters[0]"] + df["BoxCenters[1]"] - df["BoxWidths[0]"] / 2 - df["BoxWidths[1]"] / 2) / 2
    df["BoxRight0"] = (df["BoxCenters[0]"] + df["BoxCenters[1]"] + df["BoxWidths[0]"] / 2 + df["BoxWidths[1]"] / 2) / 2
    df["BoxLeft8"] = (df["BoxLeft"] + df["BoxLeftPrev"]) / 2
    df["BoxRight8"] = (df["BoxRight"] + df["BoxRightPrev"]) / 2
    df["HorizontalMovement"] = df.apply(lambda x: CalculateHorizontalMovement(x["BoxLeft8"], x["BoxRight8"],
                                                                              x["BoxLeft0"], x["BoxRight0"]), axis=1)
    df["VerticalMovement"] = (df["BoxBottoms[0]"] + df["BoxBottoms[1]"])
    df["MovementRadius"] = df["VerticalMovement"] * df["VerticalMovement"] + \
                           df["HorizontalMovement"] * df["HorizontalMovement"]
    df["MovementAngle"] = df.apply(lambda x: math.atan2(x["VerticalMovement"], x["HorizontalMovement"]), axis=1)

    del (df["BoxLeft0"], df["BoxRight0"], df["BoxLeft8"], df["BoxRight8"])
    del (df["BoxRightPrev"], df["BoxRight"], df["BoxLeft"], df["BoxLeftPrev"])
    del (df["temp"])

    df["BBoxX"] = df.apply(
        lambda x: get_vec(x["BoxCenters[0]"], x["BoxCenters[1]"], x["BoxCenters[2]"], x["BoxCenters[3]"],
                          x["BoxCenters[4]"], x["BoxCenters[5]"], x["BoxCenters[6]"], x["BoxCenters[7]"],
                          x["Counter"], x["Angle"], 50), axis=1)

    df["BBoxY"] = df.apply(
        lambda x: get_vec(x["BoxBottoms[0]"], x["BoxBottoms[1]"], x["BoxBottoms[2]"], 100 * x["BoxBottoms[3]"],
                          x["BoxBottoms[4]"], x["BoxBottoms[5]"], x["BoxBottoms[6]"], 100 * x["BoxBottoms[7]"],
                          x["Counter"], x["AbsAngle"]), axis=1)

    df["X"] = df.apply(
        lambda x: get_Xcoordinate(x["BoxHeights[0]"], x["BoxHeights[1]"], x["BoxHeights[2]"], x["BoxHeights[3]"],
                                  x["BoxHeights[4]"], x["BoxHeights[5]"], x["BoxHeights[6]"], x["BoxHeights[7]"],
                                  abs(np.sin(x["AbsAngle"]))), axis=1)
    df["Y"] = df.apply(
        lambda x: get_Xcoordinate(x["BoxHeights[0]"], x["BoxHeights[1]"], x["BoxHeights[2]"], x["BoxHeights[3]"],
                                  x["BoxHeights[4]"], x["BoxHeights[5]"], x["BoxHeights[6]"], x["BoxHeights[7]"],
                                  abs(np.cos(x["AbsAngle"]))), axis=1)

    df["curvature"] = df.apply(lambda x: get_curvature(x["BBoxX"], x["BBoxY"]) / x["Counter"], axis=1)
    df["curvatureXY"] = df.apply(lambda x: pow(10, 14) * get_curvature(x["X"], x["Y"]) / x["Counter"], axis=1)
    #
    # plt.figure(figsize=(100, 50))
    # df.fillna(-1, inplace=True)
    # plot_intersection_df(df, "curvature", 0.1)
    # plt.savefig('curvature.png')
    # plt.close()
    #
    # plt.figure(figsize=(100, 50))
    # df.fillna(-1, inplace=True)
    # plot_intersection_df(df, "curvatureXY", 0.1)
    # plt.savefig('curvatureXY.png')
    # plt.close()
    # del (df["X"], df["Y"], df["BBoxY"], df["BBoxX"])
    return df


def export_features(feature_Vec, original_columns, dtypes):
    File = open('Features.txt', 'w')
    for feature in feature_Vec:
        if feature == 'Label':
            continue
        feature_to_export = feature
        if feature in original_columns:
            feature_to_export = "Values." + feature
        if feature in \
                ["Truck", "Bus", "Car", "MotorCycle", 'SideDetected', 'LaneSplitting', 'RearDetected', 'FrontDetected',
                 'InTheSameLane']:
            feature_to_export = "r32(" + feature_to_export + ")"
        feature_to_export = feature_to_export.replace("/", "Over")
        print(feature_to_export + ",", file=File)
    File.close()


def plot_intersection_df(df, feature, quant=0.05):
    return plot_intersection(df[feature][df['Label'] == 0], df[feature][df['Label'] == 1], feature, quant)


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

    plt.plot(x, kde0_x, color='b', Label='False')
    plt.fill_between(x, kde0_x, 0, color='b', alpha=0.2)
    plt.plot(x, kde1_x, color='orange', Label='True')
    plt.fill_between(x, kde1_x, 0, color='orange', alpha=0.2)
    plt.plot(x, inters_x, color='r')
    plt.fill_between(x, inters_x, 0, facecolor='none', edgecolor='r', hatch='xx', Label='intersection')

    area_inters_x = np.trapz(inters_x, x)

    handles, Labels = plt.gca().get_legend_handles_Labels()
    Labels[2] += f': {area_inters_x * 100:.1f} %'
    plt.legend(handles, Labels)
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


def get_series_ratio(data_series):
    epsilon = 1e-10  # Assuming epsilon value, you can adjust as needed

    if len(data_series) == 1:
        return 1.0
    elif len(data_series) == 2:
        return data_series[1] / (data_series[0] + epsilon)
    elif len(data_series) == 3:
        return data_series[2] / (data_series[0] + epsilon)
    elif len(data_series) == 4:
        return (data_series[3] + data_series[2]) / (data_series[1] + data_series[0] + epsilon)
    elif len(data_series) == 5:
        return (data_series[4] + data_series[3]) / (data_series[1] + data_series[0] + epsilon)
    elif len(data_series) == 6:
        return (data_series[5] + data_series[4] + data_series[3]) / (
                data_series[2] + data_series[1] + data_series[0] + epsilon)
    elif len(data_series) == 7:
        return (data_series[5] + data_series[4] + data_series[6]) / (
                data_series[2] + data_series[1] + data_series[0] + epsilon)

    return (data_series[-4] + data_series[-3] + data_series[-2] + data_series[-1]) / (
            data_series[-5] + data_series[-6] + data_series[-7] + data_series[-8])


def objective(space, X_train, X_test, y_train, y_test):
    clf = xgb.XGBClassifier(
        n_estimators=int(space['n_estimators']), max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']))

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10, verbose=False)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


def get_last_index(list):
    file = list[-1]
    file = file.split("Spot_")[-1]
    return int(file.split("_")[0])


def get_recall_per_files(df):
    filtered_df = df[~df['Black Box Filename'].str.contains("False")]
    filtered_df = filtered_df[filtered_df["ClassifierScore"] > 0.6]
    filenames = filtered_df['lack Box Filename'].unique()
    filenames = [file for file in filenames if "_Spot_Marginal" not in file]
    bike_right = [file for file in filenames if file.startswith("Bike_Right")]
    bike_left = [file for file in filenames if file.startswith("Bike_Left") and not "And_Right" in file]
    left = [file for file in filenames if file.startswith("Left") and not "And_Right" in file]
    right = [file for file in filenames if file.startswith("Right")]
    n_bike_right = get_last_index(bike_right)
    n_bike_left = get_last_index(bike_left)
    n_left = get_last_index(left)
    n_right = get_last_index(right)

    return {"Bike": (len(bike_right) + len(bike_left)) / (n_bike_left + n_bike_right),
            "Other": (len(right) + len(left)) / (n_left + n_right)}


def model_train(df, features, alert_type, params={}, true_weight=1, threshold=0.6):
    num_round = 100
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['Label'],
                                                        test_size=0.2,
                                                        random_state=23, stratify=df['Label'])
    weights = np.ones(len(y_train))
    weights[y_train == 1] = true_weight

    if true_weight != 1:
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)

        dtest = xgb.DMatrix(X_test, label=y_test)
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

    xgbmodel = xgb.train(params, dtrain, num_round)

    y_pred = xgbmodel.predict(dtest)
    y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
    n_falses = np.sum([y1 if y2 == 0 else 0 for y1, y2 in zip(y_pred_binary, y_test)])
    n_missed = np.sum([1 - y1 if y2 == 1 else 0 for y1, y2 in zip(y_pred_binary, y_test)])
    recall = recall_score(y_test, y_pred_binary)
    print("recall = ", recall, "precision = ", precision_score(y_test, y_pred_binary),
          "accuracy = ", accuracy_score(y_test, y_pred_binary), "F1 = ", f1_score(y_test, y_pred_binary),
          "n_points = ", len(y_pred), "n_missed = ", n_missed,
          "n_falses", n_falses, "auc", roc_auc_score(y_test, y_pred)
        )

    folder_name = "model_statistics"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    csv_filename = os.path.join(folder_name, alert_type + "_statistics.csv")

    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d")
    current_time = current_datetime.strftime("%H:%M:%S")
    # recall_files = get_recall_per_files(df)

    data = {
        "Date": current_date,
        "Time": current_time,
        "recall": recall,
        "precision": precision_score(y_test, y_pred_binary),
        "accuracy": accuracy_score(y_test, y_pred_binary),
        "auc": roc_auc_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred_binary),
        "n_points": len(y_test),
        "n_missed": n_missed,
        "n_falses": n_falses,
        "true_weight": true_weight
        # "recall_bike_file": recall_files["Bike"],
        # "recall_other_file": recall_files["Other"]
    }
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            csv_writer.writeheader()

        csv_writer.writerow(data)

    return xgbmodel

def generate_lines(filename, features):
    with open(filename, 'w') as file:
        sorted_features = sorted(features)
        for item in sorted_features:
            line = f'Model->SetFeature("{item}", Values.{item});\n'
            file.write(line)
def get_features_from_file(ModelFileName):
    with open(ModelFileName, "r") as json_file:
        model_json = json.load(json_file)
    return model_json['learner']['feature_names']


def model_statistics(filename, score_column, alert_type):
    df = pd.read_csv(filename)
    y = df["Label"]
    y_pred = df[score_column]
    y_pred_binary = [1 if p >= 0.7 else 0 for p in y_pred]
    n_falses = np.sum([y1 if y2 == 0 else 0 for y1, y2 in zip(y_pred_binary, y)])
    n_missed = np.sum([1 - y1 if y2 == 1 else 0 for y1, y2 in zip(y_pred_binary, y)])
    recall = recall_score(y, y_pred_binary)
    print("recall = ", recall, "precision = ", precision_score(y, y_pred_binary),
          "accuracy = ", accuracy_score(y, y_pred_binary), "F1 = ", f1_score(y, y_pred_binary),
          "n_points = ", len(y_pred), "n_missed = ", n_missed,
          "n_falses", n_falses)

    folder_name = "model_statistics"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    csv_filename = os.path.join(folder_name, alert_type + "_statistics.csv")

    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d")
    current_time = current_datetime.strftime("%H:%M:%S")
    # recall_files = get_recall_per_files(df)

    data = {
        "Date": current_date,  # Add the current date as the first column
        "Time": current_time,  # Add the current time as the second column
        "recall": recall,
        "precision": precision_score(y, y_pred_binary),
        "accuracy": accuracy_score(y, y_pred_binary),
        "F1": f1_score(y, y_pred_binary),
        "n_points": len(y),
        "n_missed": n_missed,
        "n_falses": n_falses,
        # "recall_bike_file": recall_files["Bike"],
        # "recall_other_file": recall_files["Other"]
    }
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            csv_writer.writeheader()

        csv_writer.writerow(data)

    return


def check_accuracy(df, videos_directory, alert_string):
    video_files = [file for file in os.listdir(videos_directory) if file.lower().find(alert_string) != -1]
    df_passed = np.unique(df[df["ClassifierResultWithRules"] > 0.5]['Black Box Filename'])
    df_passed = [file for file in df_passed if
                 file.lower().find(alert_string) != -1 and file.lower().find("false") == -1]
    missed_true = []
    for file in video_files:
        if file not in df_passed:
            missed_true.append(file)
    return missed_true
