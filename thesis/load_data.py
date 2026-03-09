import pandas as pd
from pathlib import Path

def add_dtypes(data):
    '''Returns data with dtypes correctly assigned.'''

    num_cols = ['encounter_id','patient_nbr','time_in_hospital', 'num_lab_procedures', \
                    'num_procedures', 'num_medications', 'number_outpatient', \
                        'number_emergency', 'number_inpatient', 'number_diagnoses']
    numeric_cols = [x for x in num_cols if x in data.columns]  
    categorical = data.columns.difference(numeric_cols)

    # assign dtypes to float to numeric columns
    data[numeric_cols] = data[numeric_cols].astype('float')

    # assign dtypes to object to categorical columns
    data[categorical] = data[categorical].astype('object')

    # return data
    return data

# function to load the data
def load_data(processed=True):
    '''Returns the data.
    processed: bool: type of data to load (raw or processed)
    '''
    module_dir = Path(__file__).resolve().parent
    candidate_data_dirs = [
        module_dir / 'data',          # thesis/data (expected)
        module_dir.parent / 'data',   # fallback: Code/data
        Path.cwd() / 'data'           # fallback: current working dir/data
    ]

    file_stem = 'diabetic_data_preprocessed' if processed else 'diabetic_data'
    rel_paths = [Path(f'{file_stem}.csv'), Path(file_stem)]

    data_path = None
    checked_paths = []
    for data_dir in candidate_data_dirs:
        for rel_path in rel_paths:
            candidate = data_dir / rel_path
            checked_paths.append(str(candidate))
            if candidate.exists():
                data_path = candidate
                break
        if data_path is not None:
            break

    # read the CSV (raise a clear error if missing)
    if data_path is None:
        raise FileNotFoundError(
            "Data file not found. Checked:\n- " + "\n- ".join(checked_paths)
        )

    data = pd.read_csv(str(data_path), na_values=["?"])
    return add_dtypes(data)
