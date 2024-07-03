import os.path

import pandas as pd


def get_min_id(df):
    return df.groupby(["Black Box Filename", "Alert Type"])["Id"].min()


def set_min_id(id, filename, id_index):
    try:
        return id - id_index[filename]
    except:
        return -1


def get_final_speed(speed_mean, speed_by_width, speed_by_height):
    counter = 1
    speed = speed_mean

    if abs(speed_by_width) < 60.0 and speed_by_width != 0 and (
            speed_by_width * speed_mean > 0 or speed_by_width * speed_by_height > 0):
        speed += speed_by_width
        counter += 1

    if abs(speed_by_height) < 60.0 and speed_by_height != 0 and (
            speed_by_height * speed_mean > 0 or speed_by_width * speed_by_height > 0):
        speed += speed_by_height
        counter += 1

    return speed / counter


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


def CorrectAbsSpeed(df_tagged, new_df):
    tagged_speed_df = df_tagged.loc[df_tagged['AbsSpeedKMH'] != -1]
    for ind in tagged_speed_df.index:
        filename = df_tagged["Black Box Filename"][ind]
        frame_number = df_tagged["Black Box Frame Number"][ind]
        speed = df_tagged["AbsSpeedKMH"][ind]
        new_df[(new_df["Black Box Filename"] == filename) & (new_df["Black Box Filename"] == frame_number)][
            "AbsSpeedKMH"] = speed
    return new_df


def UnsupervisedLabelFront(df, AlertType, score_column):
    df.sort_values(by=['Black Box Filename', "Id", "Black Box Frame Number"], inplace=True)
    prev_row = {"Black Box Filename": "None", "Black Box Frame Number": -1, "Id": -1}
    for index, row in df.iterrows():
        if row["Black Box Filename"] != prev_row["Black Box Filename"] and row["Id"] == prev_row["Id"] and \
                abs(int(row["Black Box Frame Number"]) - int(row["Black Box Frame Number"])) < 6:
            df.at[index, 'Black Box Filename'] = prev_row["Black Box Filename"]
        prev_row = row
    df.reset_index(inplace=True)
    if "index" in df.columns:
        del (df["index"])
    df["Label"] = 0
    if "safe" in AlertType.lower():
        df["Label"][df[score_column] > 0.2] = -1
        df["Label"][(~df["Black Box Filename"].str.contains("safe"))] = -1
        df["Label"][(df["Alert Type"] == "SafeDistance") & (df[score_column] > 0.4) & (
            df["Black Box Filename"].str.contains("Distance"))] = 1
        indices = df[(df["Alert Type"] == "SafeDistance") & (df[score_column] > 0.5) & (
            df["Black Box Filename"].str.contains("Distance"))].groupby("Black Box Filename")[score_column].idxmax()

    else:
        df["Label"][df[score_column] > 0.2] = -1
        df["Label"][(~df["Black Box Filename"].str.contains("Collision"))] = 0
        indices = df[(df[score_column] > 0.5) & (
            df["Black Box Filename"].str.contains("Collision"))].groupby("Black Box Filename")[
            score_column].idxmax()
        df["Label"][(df["Alert Type"] == "Front_Collision") & (df[score_column] > 0.4) & (
            df["Black Box Filename"].str.contains("Collision"))] = 1
    for Filename, ind in indices.items():
        Id = df["Id"][ind]
        selected_rows = df[(df["Id"] == Id) & (df["Black Box Filename"] == Filename)]
        selected_rows["Label"] = -1
        selected_rows["Label"][selected_rows[score_column] > 0.5] = 1
        label = list(selected_rows["Label"])

        if max(label) > 0:
            inds = [i for i, value in enumerate(label) if value == 1]
            delimiters = [i for i in range(1, len(inds)) if inds[i] - inds[i - 1] > 10]
            if len(delimiters) > 0:
                segment_sizes = [delimiters[0]] + [delimiters[i] - delimiters[i - 1] for i in
                                                   range(1, len(delimiters))] + [len(inds) - delimiters[-1]]
                position_of_maximum = segment_sizes.index(max(segment_sizes))
                if position_of_maximum == 0:
                    inds = inds[: delimiters[position_of_maximum]]
                elif position_of_maximum == len(delimiters):
                    inds = inds[delimiters[position_of_maximum - 1]:]

                else:
                    inds = inds[delimiters[position_of_maximum - 1]: delimiters[position_of_maximum]]

            for i in range(inds[0], inds[-1]):
                label[i] = 1

        selected_rows["Label"] = label
        selected_rows["Label"][selected_rows[score_column] < 0.2] = -1
        df.loc[selected_rows.index, "Label"] = selected_rows["Label"]
    for i in range(1, len(df) - 2):
        if df["Label"][i] == -1 and df["Label"][i - 1] == 1 and (df["Label"][i + 1] == 1 or df["Label"][i + 2] == 1):
            df["Label"][i] = 1
    df.loc[(df["Alert Type"] == "Front_Collision") & (
        df["Black Box Filename"].str.contains("Marginal_Collision")), "Label"] = -1
    df['Label'][df['Black Box Filename'].str.contains('False')] = 0
    return df


def update_label(row, max_score, max_id):
    if row['Label'] == 0 and (max_score.get(row['Id'], 0) > 0.6 or max_id.get(row['Id'], 0) == 1):
        return -1
    return row['Label']


def UnsupervisedLabelBack(df, ScoreColumn="ClassifierScore"):
    df.sort_values(by=['Black Box Filename', "Id", "Black Box Frame Number"], inplace=True)
    df.reset_index(inplace=True)
    prev_row = {"Black Box Filename": "None", "Black Box Frame Number": -1, "Id": -1}
    for index, row in df.iterrows():
        if row["Black Box Filename"] != prev_row["Black Box Filename"] and row["Id"] == prev_row["Id"] and \
                abs(int(row["Black Box Frame Number"]) - int(row["Black Box Frame Number"])) < 6:
            df.at[index, 'Black Box Filename'] = prev_row["Black Box Filename"]
        prev_row = row

    ids = df["Id"]
    prev_id = -1
    for i, id in enumerate(ids):
        if id < prev_id and id < 1000:
            ids[i:] = ids[i:] + max(ids[:i])
        prev_id = id
        pass
    df["Id"] = ids
    if "index" in df.columns:
        del (df["index"])

    df["Label"] = 0
    df["Label"][df[ScoreColumn] > 0.1] = -1
    df["Label"][(df["Alert Type"] == "Right BlindSpot") & (df[ScoreColumn] > 0.2) & (df["Class"] == 3) & (
        df["Black Box Filename"].str.contains("Right")) & (df["Black Box Filename"].str.contains("Bike"))] = 1
    df["Label"][(df["Alert Type"] == "Left BlindSpot") & (df[ScoreColumn] > 0.2) & (df["Class"] == 3) & (
        df["Black Box Filename"].str.contains("Left")) & (df["Black Box Filename"].str.contains("Bike"))] = 1
    df["Label"][(df["Alert Type"] == "Left BlindSpot") & (df[ScoreColumn] > 0.3) & (
        df["Black Box Filename"].str.contains("Left"))] = 1
    df["Label"][(df["Alert Type"] == "Right BlindSpot") & (df[ScoreColumn] > 0.3) & (
        df["Black Box Filename"].str.contains("Right"))] = 1
    df["Label"][((df["Alert Type"] == "Right BlindSpot") | (df["Alert Type"] == "Left BlindSpot")) & (df["BBoxDistanceToOuterSide"] < 3) & (df["Label"] == 1) & (df[ScoreColumn] < 0.7)] = -1
    df["Label"][((df["Alert Type"] == "Right BlindSpot") | (df["Alert Type"] == "Left BlindSpot")) & (df["BBoxDistanceToOuterSide"] < 1) & (df["Label"] == 1)] = -1

    df["Label"][(((df[ScoreColumn] < 0.2) | (df["Label"] == -1)) | (df["DistanceXFinal"] > 5)) & (
            (df["Black Box Filename"].str.contains("Faraway")) | (
        df["Black Box Filename"].str.contains("FarAway")))] = 0

    df_temp = df[["Label", "Black Box Frame Number", "Id"]]
    for i in range(1, len(df_temp) - 2):
        if df_temp["Label"][i] != 1 and df_temp["Label"][i + 1] == 1 and df["Id"][i] == df["Id"][i + 1] and \
                abs(df_temp["Black Box Frame Number"][i] - df_temp["Black Box Frame Number"][i + 1]) < 3:
            df_temp["Label"][i] = 1
            continue

        if df_temp["Label"][i] != -1:
            continue

        if df_temp["Label"][i - 1] != -1 and \
                (df_temp["Label"][i + 1] == df_temp["Label"][i - 1] or df_temp["Label"][i + 2] == df_temp["Label"][
                    i - 1]):
            df_temp["Label"][i] = 1

        if df_temp["Label"][i - 1] == 0 and (df_temp["Label"][i + 1] == 0 or df_temp["Label"][i + 2] == 1):
            if df_temp["Id"][i] == df_temp["Id"][i - 1] and df_temp["Id"][i] == df_temp["Id"][i + 1]:
                df_temp["Label"][i] = 0

    df["Label"] = df_temp["Label"]

    result_df = df.groupby('Id')[ScoreColumn].agg(max).reset_index()

    max_score = dict(zip(result_df['Id'], result_df[ScoreColumn]))
    result_df = df.groupby('Id')['Label'].agg(max).reset_index()

    max_id = dict(zip(result_df['Id'], result_df['Label']))

    df['Label'] = df.apply(lambda x: update_label(x, max_score, max_id), axis=1)
    mask = (df["Alert Type"] == "Left BlindSpot") & (~df["Black Box Filename"].str.contains("Left"))
    df.loc[mask, "Label"] = 0
    mask = (df["Alert Type"] == "Right BlindSpot") & (~df["Black Box Filename"].str.contains("Right"))
    df.loc[mask, "Label"] = 0
    df.loc[df['Black Box Filename'].str.contains('False'), 'Label'] = 0
    df["Label"][(df["Class"] != 3) & (df["Class"] != -1) & (df["Black Box Filename"].str.startswith("Bike"))] = 0
    df.loc[df['Black Box Filename'].str.contains('False'), 'Label'] = 0
    return df


def normalize_id(df, ScoreColumn='ClassifierScore'):
    if ScoreColumn not in df.columns:
        ScoreColumn = 'BlindSpotScore'
    min_id = df[df[ScoreColumn] > 0.3].groupby(["Black Box Filename", "Alert Type"])["Id"].min().reset_index()
    min_id["Id_min"] = min_id["Id"]

    del (min_id["Id"])
    if "Id_min" in df.columns:
        del (df["Id_min"])

    df = pd.merge(df, min_id, how='left', on=['Black Box Filename', "Alert Type"])
    df["IdPerVideo"] = df["Id"] - df["Id_min"]
    return df

def LoadAndCorrectFileNames(filename):
    df = pd.read_csv(filename)
    df_temp = df[["Black Box Filename", "Black Box Frame Number", "Id"]]
    prev_row = {"Black Box Filename": "None", "Black Box Frame Number": -1, "Id": -1}

    for index, row in df_temp.iterrows():
        if row["Black Box Filename"] != prev_row["Black Box Filename"] and row["Id"] == prev_row["Id"] and \
                abs(int(row["Black Box Frame Number"]) - int(row["Black Box Frame Number"])) < 6:
            df.at[index, 'Black Box Filename'] = prev_row["Black Box Filename"]
        prev_row = row
    return df


def add_missing_frames(frame_numbers):
    intermediate_terms = []
    if len(frame_numbers) == 1:
        return frame_numbers
    for i in range(len(frame_numbers) - 1):
        diff = abs(frame_numbers[i] - frame_numbers[i + 1])
        if diff < 3:
            for j in range(frame_numbers[i] + 1, frame_numbers[i + 1]):
                intermediate_terms.append(j)
    frame_numbers = frame_numbers + intermediate_terms
    return sorted(frame_numbers)


def ManualLabel(df_new, tagged_data_filename, score_column):
    df_tagged = pd.read_csv(tagged_data_filename)
    df_tagged = df_tagged[(~df_tagged["Black Box Filename"].str.contains("False"))]
    df_new = normalize_id(df_new, ScoreColumn=score_column)
    df_tagged = normalize_id(df_tagged, ScoreColumn=score_column)
    df_tagged["IdPerVideo"].fillna(0, inplace=True)
    grouped = df_tagged.groupby(["Black Box Filename", "Id", "Label", "Alert Type"])
    df_new["Label2"] = -2
    for filename, group_df in grouped:
        df_temp = df_new[(df_new["Black Box Filename"] == group_df.iloc[0]["Black Box Filename"]) &
                         (df_new["Class"].isin(group_df["Class"])) &
                         (df_new["Alert Type"].isin(group_df["Alert Type"]))]
        orig_frame_numbers = list(group_df["Black Box Frame Number"])
        frame_numbers = add_missing_frames(orig_frame_numbers)
        label = group_df["Label"].unique()[0]
        id = group_df["Id"].unique()[0]
        id_per_video = group_df["IdPerVideo"].unique()[0]
        frame_numbers_used = []
        flag = False
        # print("Tagging", filename)
        for i, row in df_temp.iterrows():
            if row["Black Box Frame Number"] in frame_numbers and (row["Id"] == id or
                                                                   row["IdPerVideo"] == id_per_video or
                                                                   len(df_temp["Id"].unique()) == 1):
                if row["Black Box Frame Number"] in frame_numbers_used:
                    print(group_df.iloc[0]["Black Box Filename"], group_df.iloc[0]["Black Box Frame Number"],
                          "Id = ", group_df.iloc[0]["Id"], "Class = ", group_df.iloc[0]["Class"],
                          group_df.iloc[0]["Label"], "double tag")
                frame_numbers_used.append(row["Black Box Frame Number"])
                df_new.loc[i, "Label2"] = label
                flag = True
        if not flag:
            print(group_df.iloc[0]["Black Box Filename"], group_df.iloc[0]["Black Box Frame Number"],
                  group_df.iloc[0]["Id"], group_df.iloc[0]["Label"], "not found")
    df_new["Label"][df_new["Label2"] != -2] = df_new["Label2"]

    for i in range(1, len(df_new) - 2):
        if df_new.loc[i - 1, "Id"] == df_new.loc[i + 2, "Id"] and \
                df_new.loc[i - 1, "Label"] == df_new.loc[i + 2, "Label"] and \
                df_new.loc[i - 1, "Label"] != df_new.loc[i, "Label"]:
            df_new.loc[i, "Label"] = df_new.loc[i + 2, "Label"]

        if df_new.loc[i - 1, "Id"] == df_new.loc[i + 1, "Id"] and \
                df_new.loc[i - 1, "Label"] == df_new.loc[i + 1, "Label"] and \
                df_new.loc[i - 1, "Label"] != df_new.loc[i, "Label"]:
            df_new.loc[i, "Label"] = df_new.loc[i + 1, "Label"]
    # df_new["label"] = df_new["Label"]
    return df_new


def LabelData(new_data_filename, alert_type, tagged_data_filename=None):
    if tagged_data_filename is None:
        tagged_data_filename = new_data_filename.replace(".csv", "_Tagged.csv")
        tagged_data_filename = os.path.join("Tagged_Data", tagged_data_filename)
    score_column = 'ClassifierScore'
    # if 'Collision' in alert_type:
    #     score_column = 'CollisionScore'

    df = LoadAndCorrectFileNames(new_data_filename)

    for col in df.columns:
        if "unnamed" in col.lower():
            del (df[col])
    if "blind" not in alert_type.lower():
        df = UnsupervisedLabelFront(df, alert_type, score_column)
        if os.path.exists(tagged_data_filename):
            # df_tagged = pd.read_csv(tagged_data_filename)
            # df_new = normalize_id(df_new, ScoreColumn=score_column)
            # df_tagged = normalize_id(df_tagged)
            # df_tagged["LabelPrevious"] = df_tagged["Label"]
            # df_new = pd.merge(df_new, df_tagged[
            #     ['Black Box Filename', "Alert Type", "IdPerVideo", "Black Box Frame Number", "LabelPrevious"]],
            #                   how='left',
            #                   on=['Black Box Filename', "Alert Type", "IdPerVideo", "Black Box Frame Number"])
            # df_new['LabelPrevious'] = df_new.groupby('Id')['LabelPrevious'].ffill()
            #
            # sequences = df_new['LabelPrevious'].ne(df_new['LabelPrevious'].shift()) | df_new['LabelPrevious'].ne(
            #     df_new['LabelPrevious'].shift(-1))
            #
            # for id_val, group_df in df_new.groupby('Id'):
            #     for _, sequence_df in group_df.groupby(sequences):
            #         if len(sequence_df) > 1 and sequence_df['LabelPrevious'].nunique() == 1:
            #             fill_value = sequence_df['LabelPrevious'].iloc[0]
            #             df_new.loc[sequence_df.index, 'LabelPrevious'] = fill_value
            #
            # df_new["Label"] = df_new["LabelPrevious"].fillna(df_new["Label"])
            df = ManualLabel(df, tagged_data_filename, score_column)
    else:
        df = UnsupervisedLabelBack(df)
        if os.path.exists(tagged_data_filename):
            df = ManualLabel(df, tagged_data_filename, score_column)
    return df


def merge_speeds(df, df_speeds):
    del (df_speeds["text"])
    df_new = pd.merge(df, df_speeds,
                      how='left', on=['Black Box Filename', "Black Box Frame Number"])
    # df_new.to_csv("FrontAlertsSpeeds.csv")
    return df_new
