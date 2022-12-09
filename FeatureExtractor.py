import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import math


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
    random_state = 123
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
    df_passed = [file for file in df_passed if
                 file.lower().find(alert_string) != -1 and file.lower().find("false") == -1]
    missed_true = []
    for file in video_files:
        if file not in df_passed:
            missed_true.append(file)
    return missed_true
