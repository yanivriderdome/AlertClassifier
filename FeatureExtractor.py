import numpy as np

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
        k +=  abs(dX[i + 1] * ddY[i] - ddX[i] * dY[i + 1]) / pow(denominator, 3 / 2)
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
        k +=  abs(dX[i + 1] * ddY[i] - ddX[i] * dY[i + 1]) / pow(denominator, 3 / 2)
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
    df["curvatureXY"] = df.apply(lambda x: pow(10,14) * get_curvature(x["X"], x["Y"]) / x["Counter"], axis=1)
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
