import pandas as pd
import numpy as np


def BasicLabel(vehicle_class, Filename, label):
    if Filename.lower().find("false") != -1:
        return 0
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
    return label


def get_min_id(df):
    return df.groupby("Black Box Filename")["Id"].min()


def set_min_id(id, filename, id_index):
    return id - id_index[filename]


def get_false_ids(df):
    df_temp = df.copy()
    df_temp["lower"] = df_temp['Black Box Filename'].map(lambda x: x.lower())
    df_temp = df_temp[df_temp["lower"].map(lambda x: x.find("false") == -1)]
    new_id = df_temp.groupby(['Black Box Filename', "id_per_video"])["Label"].max()
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
    tagged_speed_df = df_tagged.loc[df_tagged['AbsoluteSpeedKMH'] != -1]
    for ind in tagged_speed_df.index:
        filename = df_tagged["Black Box Filename"][ind]
        frame_number = df_tagged["Black Box Frame Number"][ind]
        speed = df_tagged["AbsoluteSpeedKMH"][ind]
        new_df[(new_df["Black Box Filename"] == filename) & (new_df["Black Box Filename"] == frame_number)]["AbsoluteSpeedKMH"] = speed
    return new_df


def LabelData(new_data_folder, tagged_data_folder=None):
    if tagged_data_folder is None:
        tagged_data_folder = new_data_folder.replace(".csv", "_tagged.csv")
    df_new = pd.read_csv(new_data_folder)
    df_new.sort_values(by=['Black Box Filename', "Id", "Black Box Frame Number"], inplace=True)
    df_tagged = pd.read_csv(tagged_data_folder)
    # df_new["Label"] = -2
    df_new["label"] = df_new.apply(lambda x: BasicLabel(x["Class"], x["Black Box Filename"], x["Label"]), axis=1)

    min_id_new = get_min_id(df_new)
    df_new["id_per_video"] = df_new.apply(lambda x: set_min_id(x["Id"], x["Black Box Filename"], min_id_new), axis=1)
    min_id = get_min_id(df_tagged)
    df_tagged["id_per_video"] = df_tagged.apply(lambda x: set_min_id(x["Id"], x["Black Box Filename"], min_id), axis=1)
    false_true_vehicle_list = get_false_ids(df_tagged)
    mixed_vehicle_list = get_mixed_ids(df_tagged)

    df_new["label"] = retag_false_trues(df_new, false_true_vehicle_list)
    df_new["label"] = retag_mixed_trues(df_new, mixed_vehicle_list, false_true_vehicle_list)
    del (df_new["Label"])
    df_merged = pd.merge(df_new, df_tagged[['Black Box Filename', "id_per_video", "Black Box Frame Number", "Label"]],
                         how='left', on=['Black Box Filename', "id_per_video", "Black Box Frame Number"])
    df_merged.apply(lambda x: min_between_columns(x["label"], x["Label"]), axis=1)
    df_new.rename(columns={"label": "Label"}, inplace=True)
    cols = df_new.columns.tolist()
    ind2 = np.where(np.asarray(cols) == "Id")[0][0] + 1
    cols2 = cols[:ind2] + ["Label"] + cols[ind2:]
    ind1 = np.where(np.asarray(cols) == "Label")[0][0]
    del (cols2[ind1])
    df_new = df_new[cols2]
    df_new.to_csv(new_data_folder.replace(".csv", "_tagged_NEW.csv"), index=False)
    df_new = CorrectAbsSpeed(df_tagged, df_new)
    return df_new
