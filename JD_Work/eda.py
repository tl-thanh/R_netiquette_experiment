# Imports
import numpy as np
import pandas as pd

# Global variables
TREATMENT_OUTCOMES = [f'Q20_{x}' for x in range(1, 17)]
CONTROL_OUTCOMES = [f'Q9_{x}' for x in range(1, 17)]
ALL_OUTCOMES = CONTROL_OUTCOMES + TREATMENT_OUTCOMES
DEMOGRAPHICS = [f'Q{x}' for x in range(1, 8)]

# let's formalize our data cleaning into functions
def drop_first_two_rows(df):
    """Remove the first two rows that are duplicates of column headers"""
    print(f"Dropping first two header/metadata rows")
    return df.drop([0, 1])

def drop_empty_values(df, target_cols):
    """Remove rows that have null values in the target columns
    
    Parameters:
        df (pd.DataFrame): dataframe
        target_cols (list of str): column names to search for null values
    """
    dense_df = df.dropna(axis=0, subset=target_cols)
    rows_dropped = df.shape[0] - dense_df.shape[0]
    print(f"Dropping null values in {target_cols}: {rows_dropped}")
    return dense_df

def drop_unfinished_surveys(df):
    """Remove rows that have less than 100% Progress and are unfinished"""
    unfinished_surveys = df.loc[df.Progress.apply(lambda x: int(x)) < 100].index
    rows_dropped = unfinished_surveys.shape[0]
    print(f"Dropping surveys with less than 100% progress: {rows_dropped}")
    return df.drop(unfinished_surveys)

def drop_not_paying_attention(df):
    """Remove rows that did not mark attention question X_16 as 'Strongly Disagree'"""
    bad_answer_index = df.loc[(df.Q9_16 != "Strongly Disagree") & (df.Q20_16 != "Strongly Disagree")].index
    rows_dropped = bad_answer_index.shape[0]
    print(f"Dropping surveys that didn't mark attention question correctly: {rows_dropped}")
    return df.drop(bad_answer_index)

def map_likert_scores(df, outcome_cols=ALL_OUTCOMES):
    """Map survey responses to ordinal likert scores"""
    likert_scores = {"Strongly Disagree": 1,
                  "Disagree": 2,
                  "Somewhat disagree": 3,
                  "Neither agree nor disagree": 4,
                  "Somewhat agree": 5,
                  "Agree": 6,
                  "Strongly agree": 7
                 }
    df.loc[:, outcome_cols] = df.loc[:, outcome_cols].replace(likert_scores)
    return df

def get_questions(df, target_cols):
    """Return a subset of the data with participant answers to target question columns
    
    Parameters:
        df (pd.DataFrame): dataframe
        target_cols (list of str): column names to subset on
    """
    print(f"Slicing non-null values for {target_cols}")
    return df.loc[:, target_cols].dropna(axis=0, how='all')

def one_hot_race(df):
    """Unroll the race categorical variables into multiple one-hot encodings"""
    # first, get all one-hots
    onehot = pd.get_dummies(df.Q3)
    
    # next, identify the columns with ',' the indicate more than one race
    multiple_race_columns = onehot.filter(like=',', axis=1).columns
    
    # DECISION: answer that marked both 'white' and 'DTS' is suspicious, defaulting to DTS
    if "White,Decline to state" in multiple_race_columns:
        fix_white_dts = onehot[onehot.loc[:, "White,Decline to state"] == 1].index
        onehot.loc[fix_white_dts, "Decline to state"] = 1
    
    # for each column in the multiple race columns, split on commas into new column list and mark those values as 1
    for col in multiple_race_columns:
        multiple_race_cols = col.split(',')
        multiple_race_idx = onehot[onehot.loc[:, col] == 1].index
        onehot.loc[multiple_race_idx, multiple_race_cols] = 1
    
    # drop the multiple race columns
    onehot = onehot.drop(multiple_race_columns, axis=1)
    
    # drop the Q3 column from the dataframe and replace it with the onehot race column
    df = df.drop("Q3", axis=1)
    return pd.concat([df, onehot], axis=1)

def one_hot_employment(df):
    """Unroll the employment categorical variables into multiple one-hot encodings"""
    # follows protocol for race
    onehot = pd.get_dummies(df.Q6)
    multiple_employment_columns = onehot.filter(like=',', axis=1).columns
    for col in multiple_employment_columns:
        multiple_employment_cols = col.split(',')
        multiple_employment_idx = onehot[onehot.loc[:, col] == 1].index
        onehot.loc[multiple_employment_idx, multiple_employment_cols] = 1
    
    onehot = onehot.drop(multiple_employment_columns, axis=1)
    df = df.drop("Q6", axis=1)
    return pd.concat([df, onehot], axis=1)

def fix_unanswered_regions(df):
    """Replace '-99' with 'Decline to state'"""
    df.loc[:, 'Q7'] = df.loc[:, 'Q7'].replace({'-99': 'Decline to state'})
    return df

def clean_data(df):
    """Perform end-to-end loading, cleaning, and splitting of data"""
    # First, we drop the first two header/metadata rows
    df = drop_first_two_rows(df)

    # Second, we drop rows that have null values in critical data columns
    df = drop_empty_values(df, target_cols=['Progress'])

    # Third, we drop surveys that haven't been 100% completed (attrition)
    df = drop_unfinished_surveys(df)

    # Fourth, we drop surveys where participants were not paying attention and marked QX_16 as "Strongly Disagree"
    df = drop_not_paying_attention(df)

    # Fifth, we map evaluation answers to a numeric ordinal Likert Score
    df = map_likert_scores(df)

    # Sixth, we clean and encode demographic information as potential covariates
    df = one_hot_race(df)
    df = one_hot_employment(df)
    df = fix_unanswered_regions(df)

    # Seventh, we produce a dataframe that has all of our variables of interest
    # Rename columns of interest
    new_column_names = {
        'Q1': 'pronouns',
        'Q2': 'orientation',
        'Q4': 'age',
        'Q5': 'education',
        'Q7': 'region',
        'Q9_1': 'control_friendly',
        'Q9_2': 'control_positive',
        'Q9_3': 'control_sincere',
        'Q9_4': 'control_comfortable',
        'Q9_5': 'control_work_with',
        'Q9_6': 'control_situation',
        'Q9_7': 'control_peers',
        'Q9_8': 'control_others_above',
        'Q9_9': 'control_others_below',
        'Q9_10': 'control_externally',
        'Q9_11': 'control_hardworking',
        'Q9_12': 'control_knowledgable',
        'Q9_13': 'control_motivated',
        'Q9_14': 'control_leadership',
        'Q9_15': 'control_project',
        'Q20_1': 'treatment_friendly',
        'Q20_2': 'treatment_positive',
        'Q20_3': 'treatment_sincere',
        'Q20_4': 'treatment_comfortable',
        'Q20_5': 'treatment_work_with',
        'Q20_6': 'treatment_situation',
        'Q20_7': 'treatment_peers',
        'Q20_8': 'treatment_others_above',
        'Q20_9': 'treatment_others_below',
        'Q20_10': 'treatment_externally',
        'Q20_11': 'treatment_hardworking',
        'Q20_12': 'treatment_knowledgable',
        'Q20_13': 'treatment_motivated',
        'Q20_14': 'treatment_leadership',
        'Q20_15': 'treatment_project',
    }
    df = df.rename(columns=new_column_names)

    unnecessary_columns = ['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress',
           'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
           'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
           'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'Decline to state',
           'DistributionChannel', 'UserLanguage', 'Consent', 'Q9_16', 'Q9_DO', 'Q20_16', 'Q20_DO', 'FL_16_DO']
    
    if "Random ID" in df.columns:
        unnecessary_columns.append("Random ID")
    if "Q19" and "Q19 - Topics" in df.columns:
        unnecessary_columns.extend(["Q19", "Q19 - Topics"])
    return df.drop(unnecessary_columns, axis=1)

def clean_and_split_data(df):
    """Perform end-to-end data cleaning and return treatment and control groups"""
    df = clean_data(df)
    control_idx = df.loc[~df.control_friendly.isna()].index
    treatment_idx = df.loc[~df.treatment_friendly.isna()].index
    control_cols = df.filter(like='control_')
    treatment_cols = df.filter(like='treatment_')
    control = df.loc[control_idx]
    control = control.drop(treatment_cols, axis=1)
    treatment = df.loc[treatment_idx]
    treatment = treatment.drop(control_cols, axis=1)
    return control, treatment