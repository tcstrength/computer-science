import requests
import zipfile
import pandas as pd
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from torchfm.data._fmdataset import FMDataset


ZIPCODE_MAPPING = [
    {"zipcode_start": "-", "zipcode_end": "-", "state_code": 'Unknown'},
    {"zipcode_start": "00501", "zipcode_end": "00544", "state_code": 'US-NY'},
    {"zipcode_start": "00601", "zipcode_end": "00988", "state_code": 'US-PR'},
    {"zipcode_start": "01001", "zipcode_end": "02791", "state_code": 'US-MA'},
    {"zipcode_start": "02801", "zipcode_end": "02940", "state_code": 'US-RI'},
    {"zipcode_start": "03031", "zipcode_end": "03897", "state_code": 'US-NH'},
    {"zipcode_start": "03901", "zipcode_end": "04992", "state_code": 'US-ME'},
    {"zipcode_start": "05001", "zipcode_end": "05907", "state_code": 'US-VT'},
    {"zipcode_start": "06001", "zipcode_end": "06928", "state_code": 'US-CT'},
    {"zipcode_start": "07001", "zipcode_end": "08989", "state_code": 'US-NJ'},
    {"zipcode_start": "09001", "zipcode_end": "09999", "state_code": 'US-AA'},
    {"zipcode_start": "10001", "zipcode_end": "14925", "state_code": 'US-NY'},
    {"zipcode_start": "15001", "zipcode_end": "19640", "state_code": 'US-PA'},
    {"zipcode_start": "19701", "zipcode_end": "19980", "state_code": 'US-DE'},
    {"zipcode_start": "20001", "zipcode_end": "20599", "state_code": 'US-DC'},
    {"zipcode_start": "20601", "zipcode_end": "21930", "state_code": 'US-MD'},
    {"zipcode_start": "22001", "zipcode_end": "24658", "state_code": 'US-VA'},
    {"zipcode_start": "24701", "zipcode_end": "26886", "state_code": 'US-WV'},
    {"zipcode_start": "27006", "zipcode_end": "28909", "state_code": 'US-NC'},
    {"zipcode_start": "29001", "zipcode_end": "29948", "state_code": 'US-SC'},
    {"zipcode_start": "30001", "zipcode_end": "31999", "state_code": 'US-GA'},
    {"zipcode_start": "32003", "zipcode_end": "34997", "state_code": 'US-FL'},
    {"zipcode_start": "35004", "zipcode_end": "36925", "state_code": 'US-AL'},
    {"zipcode_start": "37010", "zipcode_end": "38589", "state_code": 'US-TN'},
    {"zipcode_start": "38601", "zipcode_end": "39776", "state_code": 'US-MS'},
    {"zipcode_start": "39813", "zipcode_end": "39901", "state_code": 'US-GA'},
    {"zipcode_start": "40003", "zipcode_end": "42788", "state_code": 'US-KY'},
    {"zipcode_start": "43001", "zipcode_end": "45999", "state_code": 'US-OH'},
    {"zipcode_start": "46001", "zipcode_end": "47997", "state_code": 'US-IN'},
    {"zipcode_start": "48001", "zipcode_end": "49971", "state_code": 'US-MI'},
    {"zipcode_start": "50001", "zipcode_end": "52809", "state_code": 'US-IA'},
    {"zipcode_start": "53001", "zipcode_end": "54990", "state_code": 'US-WI'},
    {"zipcode_start": "55001", "zipcode_end": "56763", "state_code": 'US-MN'},
    {"zipcode_start": "57001", "zipcode_end": "57799", "state_code": 'US-SD'},
    {"zipcode_start": "58001", "zipcode_end": "58856", "state_code": 'US-ND'},
    {"zipcode_start": "59001", "zipcode_end": "59937", "state_code": 'US-MT'},
    {"zipcode_start": "60001", "zipcode_end": "62999", "state_code": 'US-IL'},
    {"zipcode_start": "63001", "zipcode_end": "65899", "state_code": 'US-MO'},
    {"zipcode_start": "66002", "zipcode_end": "67954", "state_code": 'US-KS'},
    {"zipcode_start": "68001", "zipcode_end": "69367", "state_code": 'US-NE'},
    {"zipcode_start": "70001", "zipcode_end": "71497", "state_code": 'US-LA'},
    {"zipcode_start": "71601", "zipcode_end": "72959", "state_code": 'US-AR'},
    {"zipcode_start": "73001", "zipcode_end": "74966", "state_code": 'US-OK'},
    {"zipcode_start": "75001", "zipcode_end": "79999", "state_code": 'US-TX'},
    {"zipcode_start": "80001", "zipcode_end": "81658", "state_code": 'US-CO'},
    {"zipcode_start": "82001", "zipcode_end": "83128", "state_code": 'US-WY'},
    {"zipcode_start": "83201", "zipcode_end": "83876", "state_code": 'US-ID'},
    {"zipcode_start": "84001", "zipcode_end": "84791", "state_code": 'US-UT'},
    {"zipcode_start": "85001", "zipcode_end": "86556", "state_code": 'US-AZ'},
    {"zipcode_start": "87001", "zipcode_end": "88441", "state_code": 'US-NM'},
    {"zipcode_start": "88901", "zipcode_end": "89883", "state_code": 'US-NV'},
    {"zipcode_start": "90001", "zipcode_end": "96162", "state_code": 'US-CA'},
    {"zipcode_start": "96701", "zipcode_end": "96898", "state_code": 'US-HI'},
    {"zipcode_start": "97001", "zipcode_end": "97920", "state_code": 'US-OR'},
    {"zipcode_start": "98001", "zipcode_end": "99403", "state_code": 'US-WA'},
    {"zipcode_start": "99501", "zipcode_end": "99950", "state_code": 'US-AK'}
]


def __download_ml_100k():
    # Step 1: Download the ZIP file
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_folder = "ml-100k"
    zip_file_path = "ml-100k.zip"
    extract_folder = "./__tmp"
    return_path = Path(extract_folder).joinpath(zip_folder)

    if Path(zip_file_path).exists():
        return return_path

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {zip_file_path}")
    with open(zip_file_path, "wb") as file:
        with progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                if not chunk: continue
                file.write(chunk)
                progress_bar.update(len(chunk))

    # Step 2: Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    logger.info(f"Downloaded to {return_path.resolve()}.")
    return return_path


def __zipcode_to_state(zipcode: str):
    for item in ZIPCODE_MAPPING:
        if (zipcode >= item["zipcode_start"]
            and zipcode <= item["zipcode_end"]):
            return "state::" + item["state_code"]
    return "state::unknown"


def __concat_genres(row):
    items = []
    for col in row.keys():
        if not col.startswith("genre::"): continue
        if int(row[col]) == 1:
            items.append(col)
    return items


def __build_user_features(row):
    features = [row["gender"], row["occupation"], row["age_group"], row["us_state"]]
    return features


def __build_item_features(row):
    features = [row["year_group"]] + row["genres"]
    return features


def __extract_features(info: pd.DataFrame, id_col: str):
    features = info[id_col].unique().tolist()
    for _, item in info.iterrows():
        features.extend(item["features"])
    features = list(set(features))
    features = {f: i for i, f in enumerate(features)}
    return features


def load(train: bool=True) -> FMDataset:
    """Download and process to 3 pd.DataFrames, user_info, item_info and ratings.
    Returns:
        dict[str, DataFrame]: user_info, item_info, ratings, user_features, item_features
    """
    
    dir = __download_ml_100k()
    if train:
        ratings = pd.read_csv(dir.joinpath("u1.base"), sep="\t", names=[
            "user_id", "movie_id", "rating", "timestamp"
        ])
    else:
        ratings = pd.read_csv(dir.joinpath("u1.test"), sep="\t", names=[
            "user_id", "movie_id", "rating", "timestamp"
        ])

    ratings = ratings[["user_id", "movie_id", "rating"]]

    user_info = pd.read_csv(dir.joinpath("u.user"), sep="|", names=[
        "user_id", "age", "gender", "occupation", "zipcode"
    ])

    item_info = pd.read_csv(dir.joinpath("u.item"), sep="|", names=[
        "movie_id", "movie_title",   "release_date", "video_release_date", "idbm", "genre::unknown",
        "genre::action", "genre::adventure", "genre::animation", "genre::children", "genre::comedy",
        "genre::crime", "genre::documentary", "genre::drama", "genre::fantasy", "genre::film-noir",
        "genre::horror", "genre::musical", "genre::mystery", "genre::romance", "genre::sci-fi",
        "genre::thriller", "genre::war", "genre::western"
    ], encoding="latin1")


    age_bins = [0, 18, 25, 35, 45, 55, 100]
    year_bins = list(range(1920, 2001, 10))
    user_info["age_group"] = pd.cut(user_info["age"], bins=age_bins, labels=age_bins[:-1])
    user_info["age_group"] = "age_group::" + user_info["age_group"].astype(str)
    user_info["us_state"] = user_info["zipcode"].apply(__zipcode_to_state)
    user_info["gender"] = "gender::" + user_info["gender"]
    user_info["occupation"] = "occupation::" + user_info["occupation"]
    user_info = user_info.drop(columns=["age", "zipcode"])
    user_info["features"] = user_info.apply(__build_user_features, axis=1)
    user_info = user_info[["user_id", "features"]]

    item_info["year"] = pd.to_datetime(item_info["release_date"]).dt.year
    item_info["year_group"] = pd.cut(item_info["year"], bins=year_bins, labels=year_bins[:-1])
    item_info["year_group"] = "year::" + item_info["year_group"].astype(str)
    item_info = item_info.drop(columns=["movie_title", "release_date", "video_release_date", "idbm"])
    item_info["genres"] = item_info.apply(__concat_genres, axis=1)
    ignore_cols = [x for x in item_info.columns if x.startswith("genre::")]
    item_info = item_info.drop(columns=ignore_cols)
    item_info["features"] = item_info.apply(__build_item_features, axis=1)
    item_info = item_info[["movie_id", "features"]]

    dataset = FMDataset(
        interactions=[tuple(row) for row in ratings.itertuples(index=False, name=None)],
        user_info=[tuple(row) for row in user_info.itertuples(index=False, name=None)],
        item_info=[tuple(row) for row in item_info.itertuples(index=False, name=None)],
        user_features=__extract_features(user_info, "user_id"),
        item_features=__extract_features(item_info, "movie_id")
    )
    return dataset