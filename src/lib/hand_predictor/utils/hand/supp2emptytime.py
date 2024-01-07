# import pdb

import pandas as pd


def supp2emptytimestamp(input_filepath, output_filepath=None, mode='zero', logging=False):
    '''
    input_filepath: pd.DataFrame or str [path] (xxx.csv)
    output_filepath: None (return pd.DataFrame) or str [path] (/write/to/your/path/xxx.csv)
    mode: str ["zero", "prev_frame"]
    '''
    # read input dataframe
    if type(input_filepath) == pd.DataFrame:
        df = input_filepath.copy()
    elif type(input_filepath) == str:
        df = pd.read_csv(input_filepath)
    else:
        raise NotImplementedError

    df_length = len(df)
    # pdb.set_trace()

    # if the data (keypoint) can not be extracted (empty)
    if df_length == 0:
        new_df = []
        for timestamp in range(10):
            temp = pd.Series([0] * len(df.columns), index=df.columns)
            temp['timestamp'] = timestamp
            new_df.append(temp)
        new_df = pd.DataFrame(new_df)

    # if not completely empty
    else:
        video_length = df['timestamp'].values[-1] + 1

        # if the data is perfectly converted.
        if df_length == video_length:
            new_df = df.copy()

        # deal with the data with some empty frames.
        else:
            # fill zero array
            if mode == 'zero':
                new_df = []
                df_idx = 0
                for timestamp in range(video_length):
                    # keypoint exist
                    if df['timestamp'].iloc[df_idx] == timestamp:
                        new_df.append(df.iloc[df_idx])
                        df_idx += 1

                    # keypoint not exist
                    else:
                        temp = pd.Series([0] * len(df.columns), index=df.columns)
                        temp['timestamp'] = timestamp
                        new_df.append(temp)

                new_df = pd.DataFrame(new_df)

            # fill previous timeframe
            elif mode == 'prev_frame':

                new_df = pd.DataFrame([], columns=df.columns)
                new_df_idx = 0

                for idx in range(df_length):
                    each_row = pd.DataFrame(df.iloc[idx]).T
                    video_idx = each_row['timestamp'].values
                    while (new_df_idx <= video_idx):
                        new_df = pd.concat([new_df, each_row])
                        new_df_idx += 1

                new_df.reset_index(inplace=True, drop=True)
                new_df['timestamp'] = new_df.index

            # mode typing error
            else:
                raise NotImplementedError

    # write to file
    new_df['timestamp'] = new_df['timestamp'].astype(int)
    if logging:
        print(f"Done: {input_filepath}")

    if output_filepath is not None:
        new_df.to_csv(output_filepath, index=None)
        return None

    else:
        return new_df
