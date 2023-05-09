import pandas as pd
import os


def flatten_json(y):
    out = {}

    def flatten(x, name=""):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + ".")
        elif type(x) is list:
            # Flatten only the first dictionary of the list
            if len(x) > 0 and isinstance(x[0], dict):
                for a in x[0]:
                    flatten(x[0][a], name + a + ".")
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def flatten_specific_cols(df, cols_to_flat):
    for col in cols_to_flat:
        while col in df.columns:
            if isinstance(df[col].iloc[0], (dict, list)):
                flat_col = df[col].apply(flatten_json)
                flat_col = pd.DataFrame(flat_col.tolist())
                flat_col = flat_col.add_prefix(f"{col}.")

                df = df.drop(col, axis=1)
                df = pd.concat([df, flat_col], axis=1)
            else:
                break

    return df


def load_and_flatten_json(file_path):
    df = pd.read_json(file_path, lines=True)

    cols_to_flat = ["totals", "device", "geoNetwork", "hits", "trafficSource"]

    df_flat = flatten_specific_cols(df, cols_to_flat)

    columns_to_convert = [
        # "hits.hour",
        "totals.hits",
        "totals.pageviews",
        "totals.timeOnSite",
        "totals.visits",
        "hits.latencyTracking.pageLoadTime",
    ]

    for col in columns_to_convert:
        df_flat[col] = pd.to_numeric(df_flat[col], errors="coerce")

    return df_flat


data_dir = "data/"
json_files = [
    os.path.join(data_dir, file)
    for file in os.listdir(data_dir)
    if file.endswith(".json")
]

df_combined = pd.concat(
    [load_and_flatten_json(file) for file in json_files], ignore_index=True
)

df_combined.to_pickle("data.pkl")
