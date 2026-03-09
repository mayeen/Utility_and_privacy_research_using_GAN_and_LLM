import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def process_multiple_encounters(data, drop_id_fields=True):
    """
    Keep one encounter per patient_nbr using maximum time_in_hospital.
    Optionally drop patient_nbr and race after processing.
    """
    df = data.copy()

    if "patient_nbr" in df.columns and "time_in_hospital" in df.columns:
        df = (
            df.sort_values(["patient_nbr", "time_in_hospital"], ascending=[True, False])
            .drop_duplicates(subset="patient_nbr", keep="first")
        )

    if drop_id_fields:
        df = df.drop(columns=["patient_nbr", "race"], errors="ignore")

    return df


def consolidate_age(data):
    """
    Consolidate age from 10 buckets to 3 numeric levels:
    [0-10) ... [50-60) -> 1
    [60-70), [70-80) -> 1.5
    [80-90), [90-100) -> 2
    """
    df = data.copy()
    age_map = {
        "[0-10)": 1,
        "[10-20)": 1,
        "[20-30)": 1,
        "[30-40)": 1,
        "[40-50)": 1,
        "[50-60)": 1,
        "[60-70)": 1.5,
        "[70-80)": 1.5,
        "[80-90)": 2,
        "[90-100)": 2,
    }

    if "age" in df.columns:
        df["age"] = df["age"].map(age_map)

    return df


def drop_unwanted_columns(data):
    """Remove weight, payer_code, and medical_specialty when present."""
    df = data.copy()
    return df.drop(columns=["weight", "payer_code", "medical_specialty"], errors="ignore")


def _icd9_to_group(code):
    """Map ICD-9 code into 9 diagnosis groups."""
    if pd.isna(code):
        return "Other"

    code = str(code).strip()

    if code.startswith("250"):
        return "Diabetes"

    if code.startswith(("E", "V")):
        return "Other"

    try:
        val = float(code)
    except ValueError:
        return "Other"

    if (390 <= val <= 459) or val == 785:
        return "Circulatory"
    if (460 <= val <= 519) or val == 786:
        return "Respiratory"
    if (520 <= val <= 579) or val == 787:
        return "Digestive"
    if (580 <= val <= 629) or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 800 <= val <= 999:
        return "Injury"

    return "Other"


def change_diag_columns(data):
    """Convert diag_1, diag_2, diag_3 into 9 diagnosis categories."""
    df = data.copy()
    diag_cols = ["diag_1", "diag_2", "diag_3"]

    for col in diag_cols:
        if col in df.columns:
            df[col] = df[col].apply(_icd9_to_group)

    return df


def change_medication_columns(data):
    col = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    df = data.copy()
    for x in col:
        if x in df.columns:
            df[x] = df[x].map(lambda v: 0 if v == "No" else 1)

    return df


def encode_a1cresult(data):
    """Impute and encode A1Cresult: Norm=1, >7=2, >8=3, missing=0."""
    df = data.copy()
    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].fillna(0).map({"Norm": 1, ">7": 2, ">8": 3, 0: 0})
    return df


def target_encode(df, col, mapping):
    """Label encode target column using a mapping dictionary."""
    data = df.copy()
    if col in data.columns:
        data[col] = data[col].map(mapping)
    return data


def label_encode(data, target, cat_cols=None):
    """Returns the data with categorical features labeled."""
    df = data.copy()

    if cat_cols:
        label_encoder = LabelEncoder()
        for col in cat_cols:
            if col in df.columns:
                df[col] = label_encoder.fit_transform(df[col].astype(str))
    else:
        cols = df.select_dtypes(include="object").columns

        if target in cols:
            cols = cols.drop(target)

        label_encoder = LabelEncoder()
        for col in cols:
            df[col] = label_encoder.fit_transform(df[col].astype(str))

    return df

# impute_data: Returns data with missing values imputed.
def impute_data(data):
    '''Returns data with missing values imputed.'''

    # get all categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns

    # get all numerical columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # get a copy of the data
    data_copy = data.copy()

    # impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_cat_data = imputer.fit_transform(data[cat_cols])
    data_copy.loc[:, cat_cols] = imputed_cat_data

    # impute missing values
    imputer = SimpleImputer(strategy='median')
    imputed_num_data = imputer.fit_transform(data[num_cols])
    data_copy.loc[:, num_cols] = imputed_num_data

    # return data
    return data_copy


def engineer_features(data, target="readmitted", cat_cols=None):
    """Apply requested feature engineering and encoding pipeline."""
    df = data.copy()

    # 0) keep one encounter per patient and optionally drop patient/race fields
    df = process_multiple_encounters(df)

    # 0.5) consolidate age buckets into 3 numeric levels
    df = consolidate_age(df)

    # 1) remove columns
    df = drop_unwanted_columns(df)

    # 2) change diag columns into grouped categories
    df = change_diag_columns(df)

    # 3) binary medication encoding
    df = change_medication_columns(df)

    # 4) encode A1Cresult
    df = encode_a1cresult(df)
    
    df= impute_data(df)
    # 5) label encode categorical columns (except target by default)
    df = label_encode(df, target=target, cat_cols=cat_cols)

    # 6) target encode readmission classes into binary readmission
    df = target_encode(df, target, {"NO": 0, ">30": 1, "<30": 1})



    return df
