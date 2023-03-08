import os.path

import pandas as pd
import numpy as np


def BasicLabel(vehicle_class, alert_type, Filename, label):
    if Filename.lower().find("false") != -1:
        return 0

    if vehicle_class == -1:
        return label

    if (Filename.lower().find("cycle") == -1 and Filename.lower().find("bike") == -1 and
            Filename.lower().find("truck") == -1 and Filename.lower().find("bus") == -1 and
            Filename.lower().find("car") == -1):
        return label
    if vehicle_class == 2 and Filename.lower().find("truck") == -1:
        return 0
    if vehicle_class == 3 and Filename.lower().find("bike") == -1 and Filename.lower().find("cycle") == -1:
        return 0
    if vehicle_class == 1 and Filename.lower().find("bus") == -1:
        return 0
    if vehicle_class == 0 and Filename.lower().find("car") == -1:
        return 0
    if alert_type.lower().find("safe") != -1 and Filename.lower().find("safe") == -1:
        return -1
    if alert_type.lower().find("collision") != -1 and Filename.lower().find("collision") == -1:
        return 0

    return label


def get_min_id(df):
    return df.groupby("Black Box Filename")["Id"].min()


def set_min_id(id, filename, id_index):
    return id - id_index[filename]


def get_false_ids(df):
    df_temp = df.copy()
    df_temp = df_temp[df_temp["Black Box Filename"].map(lambda x: x.lower().find("false") == -1)]
    new_id = df_temp.groupby(['Black Box Filename', 'Black Box Filename', "id_per_video"])["Label"].max()
    return new_id


def get_mixed_ids(df):
    df_temp = df.copy()
    df_temp["lower"] = df_temp['Black Box Filename'].map(lambda x: x.lower())
    df_temp = df_temp[df_temp["lower"].map(lambda x: x.find("false") == -1)]
    df_temp = df_temp[df_temp["Label"] == -1]
    new_id_frame_numbers = df_temp.groupby(['Black Box Filename', "id_per_video"])["Black Box Frame Number"].max()
    return new_id_frame_numbers


def retag(label, id_per_video, filename, false_true_vehicle_list):
    if filename in false_true_vehicle_list.keys() and id_per_video in false_true_vehicle_list[filename]:
        if false_true_vehicle_list[filename][id_per_video] == 0:
            return 0
        else:
            return -2
    return label


def retag_false_trues(df, false_true_vehicle_list):
    df.apply(lambda x: retag(x["label"], x["id_per_video"], x["Black Box Filename"], false_true_vehicle_list), axis=1)
    return df["label"]


def retag_mixed(label, id_per_video, filename, frame_number, mixed_vehicle_list, false_true_vehicle_list):
    if filename in mixed_vehicle_list.keys() and id_per_video in mixed_vehicle_list[filename]:
        if frame_number < false_true_vehicle_list[filename][id_per_video] + 3:
            return -1
    return label


def retag_mixed_trues(df, mixed_vehicle_list, false_true_vehicle_list):
    df.apply(lambda x: retag_mixed(x["label"], x["id_per_video"],
                                   x["Black Box Filename"], x["Black Box Frame Number"],
                                   mixed_vehicle_list, false_true_vehicle_list), axis=1)
    return df["label"]


def min_between_columns(label, Label_true):
    if label == Label_true:
        return label
    return Label_true


def CorrectAbsSpeed(df_tagged, new_df):
    tagged_speed_df = df_tagged.loc[df_tagged['AbsSpeedKMH'] != -1]
    for ind in tagged_speed_df.index:
        filename = df_tagged["Black Box Filename"][ind]
        frame_number = df_tagged["Black Box Frame Number"][ind]
        speed = df_tagged["AbsSpeedKMH"][ind]
        new_df[(new_df["Black Box Filename"] == filename) & (new_df["Black Box Filename"] == frame_number)][
            "AbsSpeedKMH"] = speed
    return new_df


def FixFileNames(df):
    df_shifted2 = df.shift(-1)

    df["Idm1"] = df_shifted2["Id"]
    print("Replacing the following false filenames")
    for i in [3, 2, 1]:
        df_shifted1 = df.shift(i)

        df["Filename1"] = df_shifted1["Black Box Filename"]
        df["Id1"] = df_shifted1["Id"]
        df["FrameNumber1"] = df_shifted1["Black Box Frame Number"]

        df['False Filename'] = df.apply(lambda x: (x["Filename1"] != x["Black Box Filename"] and x["Id1"] == x["Id"])
                                                  and x["Id1"] != x["Idm1"] and abs(
            x["Black Box Frame Number"] - x["FrameNumber1"]) < 3, axis=1)
        df["Black Box Filename"][df['False Filename']] = df["Filename1"][df['False Filename']]

        for x in df.index[df['False Filename']]:
            print(x, df["Black Box Filename"][x], df["Black Box Frame Number"][x], df["Id"][x])
    print("===========================================================")
    for col in ["Id1", "FrameNumber1", "Filename1", "Idm1", 'False Filename']:
        del (df[col])
    return df

def retag_false_positives(df_new, df_false_positives):
    df_false_positives["Label_new"] = df_false_positives["Label"]
    df_new["Label_corrected"] = df_new["Label"]
    del(df_false_positives["Label"])
    for i in [-1, 0, 1]:
        df_temp = df_false_positives.copy()
        df_temp["Black Box Frame Number"] = df_false_positives["Black Box Frame Number"] + i
        df_temp.drop_duplicates(inplace=True)
        # df_temp2 = df_new[['Black Box Filename', "Alert Type",  "id_per_video", "Black Box Frame Number"]]
        df_new = pd.merge(df_new, df_temp[['Black Box Filename', "Alert Type", "id_per_video", "Black Box Frame Number", "Label_new"]],
                             how='left', on=['Black Box Filename', "Alert Type",  "id_per_video", "Black Box Frame Number"])
        df_new["Label_new"].fillna(10, inplace=True)
        df_new["Label_corrected"] = df_new.apply(lambda x: min(x["Label_corrected"], x["Label_new"]) if x["Label_new"]!= 2 else x["Label_new"], axis=1)
        del(df_new["Label_new"])
    return df_new

def LabelData(new_data_filename, tagged_data_filename=None):
    if tagged_data_filename is None:
        tagged_data_filename = new_data_filename.replace(".csv", "_tagged.csv")
        tagged_data_filename = os.path.join("Tagged_Data", tagged_data_filename)

    df_tagged = pd.read_csv(tagged_data_filename)
    df_new = pd.read_csv(new_data_filename)
    # df_new.sort_values(by=['Black Box Filename', "Id", "Black Box Frame Number"], inplace=True)
    df_new["Label"] = df_new.apply(lambda x: BasicLabel(x["Class"], x["Alert Type"],
                                                        x["Black Box Filename"], x["Label"]), axis=1)

    min_id_per_video_new = get_min_id(df_new)
    df_new["id_per_video"] = df_new.apply(lambda x: set_min_id(x["Id"], x["Black Box Filename"], min_id_per_video_new),
                                          axis=1)
    min_id = get_min_id(df_tagged)
    df_tagged["id_per_video"] = df_tagged.apply(lambda x: set_min_id(x["Id"], x["Black Box Filename"], min_id), axis=1)

    df_tagged = pd.read_csv(tagged_data_filename)
    df_new = FixFileNames(df_new)
    df_new.sort_values(by=['Black Box Filename', "Id","Black Box Frame Number"], inplace=True)

    df_false_positives = df_tagged[
        df_tagged.apply(lambda x: x['Black Box Filename'].lower().find("false") == -1 and x["Label"] != 1,
                        axis=1)][['Black Box Filename', "Id", "Alert Type", "id_per_video", "Black Box Frame Number","Label"]]
    df_new = retag_false_positives(df_new, df_false_positives)
    #
    # df_false_true = get_false_ids(df_tagged)
    # mixed_vehicle_list = get_mixed_ids(df_tagged)
    # df_new["label"] = retag_false_trues(df_new, df_false_true)
    # df_new["label"] = retag_mixed_trues(df_new, mixed_vehicle_list, df_false_true)
    # # del (df_new["Label"])
    # df_merged = pd.merge(df_new, df_tagged[['Black Box Filename', "id_per_video", "Black Box Frame Number", "Label"]],
    #                      how='left', on=['Black Box Filename', "id_per_video", "Black Box Frame Number"])
    # df_merged.apply(lambda x: min_between_columns(x["label"], x["Label"]), axis=1)
    df_new.rename(columns={"label": "Label"}, inplace=True)
    cols = df_new.columns.tolist()
    ind2 = np.where(np.asarray(cols) == "Id")[0][0] + 1
    cols2 = cols[:ind2] + ["Label"] + cols[ind2:]
    ind1 = np.where(np.asarray(cols) == "Label")[0][0]
    del (cols2[ind1])
    df_new = df_new[cols2]
    df_new.to_csv(new_data_filename.replace(".csv", "_tagged_NEW.csv"), index=False)
    # df_new = CorrectAbsSpeed(df_tagged, df_new)
    return df_new


def merge_speeds(df, df_speeds):
    del(df_speeds["text"])
    df_new = pd.merge(df, df_speeds,
                      how='left', on=['Black Box Filename', "Black Box Frame Number"])
    # df_new.to_csv("FrontAlertsSpeeds.csv")
    return df_new