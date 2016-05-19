""" Water Missions International Data Manipulation

This module contains methods for retrieving and storing WMI datasets
"""

# Authors: Eric Czech

import os
import re
import pandas as pd
import numpy as np


FIELD_REGEX = re.compile('\\|###\\|')
LINE_REGEX = re.compile('\\|:::\\|')
DEFAULT_DATA_DIR = '/Users/eczech/data/research/wmi/wmi_db_20160515'


def parse_file(filepath, cols, field_regex, line_regex):

    res = []
    with open(filepath, 'rb') as fd:
        # Assume Windows encoding
        content = str(fd.read(), 'iso-8859-1')

        # Split entire string on line regex
        lines = line_regex.split(content)

        # For each line, split fields on field regex
        for i, line in enumerate(lines):
            # Ignore 0-length lines (usually this is just the last line)
            if len(line) == 0:
                continue

            # Split line into fields
            fields = field_regex.split(line)

            # Verify that the number of fields equals the number of column names
            if len(fields) != len(cols):
                msg = '(line = {}) Number of fields ({}) does not equal expected ({}):  Line content = {}'\
                    .format(i, len(fields), len(cols), line)
                raise ValueError(msg)

            # Append raw string fields to result
            res.append(fields)

    # Return data frame with fields named by given headers
    res = pd.DataFrame(res, columns=cols)
    return res.apply(lambda x: x.where(x.str.strip().str.len() > 0, None))


def load_table(table, data_dir=DEFAULT_DATA_DIR, field_regex=FIELD_REGEX, line_regex=LINE_REGEX):
    print('Loading table "{}"'.format(table))
    header_path = os.path.join(data_dir, '{}_Headers.csv'.format(table))
    data_path = os.path.join(data_dir, '{}.csv'.format(table))

    if os.path.getsize(data_path) == 0:
        print('Skipping table "{}" due to empty data file ({})'.format(table, data_path))
        return None
    assert os.path.getsize(header_path) > 0, 'Header file cannot be empty (table = {})'.format(table)

    cols = pd.read_csv(header_path, header=None, names=['col'])['col'].values
    return parse_file(data_path, cols, field_regex, line_regex)


def load_all_tables(data_dir=DEFAULT_DATA_DIR, skip_tables=['ProjectBrochures']):
    """ Load all data tables in the given data dir
    :param data_dir: Data directory continaing table data CSV file + header CSV file combinations (from MS SQL)
    :param skip_tables: Tables to ignore (due to emptiness or other formatting issues)
    :return: dict keyed by table name with data frames as values
    """
    tables = []
    for file in os.listdir(data_dir):
        tables.append(file.split('_')[0].split('.')[0])

    data = {}
    for table in np.unique(tables):
        if table in skip_tables:
            continue
        d = load_table(table, data_dir)
        if d is not None:
            data[table] = d
    return data


def search_for_column(db, column):
    tables = []
    for table, data in db.items():
        for c in data:
            if column.lower() in c.lower():
                tables.append(table)
                continue
    return list(np.unique(tables))


def get_project_summary():
    d_raw = load_table('AssessmentSummaryInformation')

    c_numeric = {
        'AssessmentID': np.int64,
        'AnticipatedPeopleServed': np.float64,
        'Population': np.float64,
        'Budget': np.float64
    }
    d = d_raw.copy()

    for c in c_numeric:
        def to_num(x):
            return pd.to_numeric(x, errors='coerce') if x else None
        d[c] = d[c].apply(to_num).astype(c_numeric[c])

    c_category = [
        'Country', 'AssessmentName', 'ProjectManager', 'Feasible',
        'Region', 'ProjectSubclassification', 'ProjectType', 'Priority'
    ]
    for c in c_category:
        d[c] = d[c].where(d[c].notnull(), 'Unknown')

    c_date = ['AssessmentCompletedOn']
    for c in c_date:
        d[c] = pd.to_datetime(d[c])

    c_non_null = ['AssessmentID', 'ProjectClassification']
    for c in c_non_null:
        d = d[d[c].notnull()]

    d = d[np.unique(c_non_null + c_category + c_date + list(c_numeric.keys()))]

    return d


def standardize_production_units(d, value_col, units_col):
    def standardize(r):
        v = pd.to_numeric(r[value_col], errors='coerce')
        u = r[units_col]
        if pd.isnull(u) or pd.isnull(v):
            return None
        u = u.lower().strip()
        if u == 'unknown':
            return None

        # Standardize on liters
        if u == 'gallons':
            return v * 3.78541
        elif u == 'liters':
            return v
        elif u == 'cubic meters':
            return v * 1000
        else:
            raise ValueError('Unit type "{}" is not supported'.format(u))
    return d.apply(standardize, axis=1)


def get_project_details():
    d_raw = load_table('ProjectsInformation')

    d_raw['OperatorCompensationMethod'] = d_raw['OperatorCompensationMethod'].apply(lambda x: x.lower() if x else None)

    c_numeric = {
        'AssessmentID': np.int64,
        'ActualHouseholds': np.float64,
        'AnticipatedBreakevenWaterPrice': np.float64,
        'ActualPeopleWithAccessToSafeWater': np.float64,
        'AnticipatedMonthlyOperationalCost': np.float64,
        'AnticipatedMonthlyReplacementCost': np.float64,
        'AnticipatedHouseholdPenetration': np.float64,
        'AnticipatedIndividualConsumption': np.float64,
        'AnticipatedDailyProduction': np.float64,
        'AverageHouseholdIncome': np.float64,
        'AverageHouseholdSize': np.float64,
        'ContainerSizeForConsumptionExpenditure': np.float64,
        'EstimatedHouseholdsInServiceArea': np.float64,
        'EstimatedAverageHouseholdSize': np.float64,
        'EstimatedPeopleInInstitutionsNotIncludedInCommunity': np.float64,
        'EstimatedHouseholdsInServiceArea': np.float64,
        'EstimatedHouseholdsInServiceArea': np.float64,
        'MaxAllowableWaterPrice': np.float64,
        'QuantityWaterDrinking': np.float64,
        'QuantityWaterCooking': np.float64,
        'QuantityWaterHygieneAndSanitation': np.float64,
        'QuantityWaterDomesticUse': np.float64,
        'QuantityWaterDrinking': np.float64,
        'QuantityWaterDrinking': np.float64,
        'TargetMonthlyOperationalCosts': np.float64,
        'TargetMonthlySavings': np.float64,
        'TargetHouseholdsCollectingWater': np.float64,
        'TargetDailyHouseholdConsumption': np.float64,
        'TargetDailyProduction': np.float64,
        'TotalPeopleServedDisasterResponse': np.float64,
        'TotalHouseholds': np.float64
    }
    d = d_raw.copy()

    for c in c_numeric:
        def to_num(x):
            return pd.to_numeric(x, errors='coerce') if x else None
        d[c] = d[c].apply(to_num).astype(c_numeric[c])

    c_category = [
        'OperatorCompensationMethod', 'NewWaterSourceToBeDeveloped', 'AverageHouseholdIncomeUnits',
        'TargetDailyProductionUnits', 'AverageHouseholdIncomeUnits', 'WaterFeeStructure'
    ]
    for c in c_category:
        d[c] = d[c].where(d[c].notnull(), 'Unknown')

    c_date = [
     'CommissioningDate', 'CommissioningScheduledDate', 'InstallationDate'
    ]
    for c in c_date:
        d[c] = pd.to_datetime(d[c])

    c_non_null = ['AssessmentID']
    for c in c_non_null:
        d = d[d[c].notnull()]

    d = d[np.unique(c_non_null + c_category + c_date + list(c_numeric.keys()))]

    # Standardize production target using volume units field
    d['TargetDailyProduction'] = standardize_production_units(d, 'TargetDailyProduction', 'TargetDailyProductionUnits')
    d = d.drop('TargetDailyProductionUnits', axis=1)

    return d


def get_country_data():
    d_cty = load_table('Countries')
    d_cty = d_cty[d_cty['Continent'].notnull()]
    d_cty = d_cty.rename(columns={
            'A2_ISO_Standard': 'CountryCode2', 'A3_UN_Standard': 'CountryCode3',
            'NetSuiteInternalID': 'CountryNetSuiteID'
    })
    d_cty = d_cty[['Country', 'Continent', 'CountryCode2', 'CountryCode3', 'CountryNetSuiteID']]
    return d_cty


# ----- Data Views ----- #


def get_project_analysis_data(rm_na=False):
    # Start with project summary data
    d_proj = get_project_summary()

    # Add country metadata
    d_cty = get_country_data()
    d_proj = pd.merge(d_proj, d_cty, on='Country')

    # Add project details
    d_proj_det = get_project_details()
    d_proj = pd.merge(d_proj, d_proj_det, how='left', on='AssessmentID')

    dt = d_proj[d_proj['ActualHouseholds'].notnull()]

    rm_cols = [
        'Budget', 'Population', 'TotalPeopleServedDisasterResponse', 'AnticipatedPeopleServed',
        'CountryCode2', 'CountryCode3', 'AssessmentCompletedOn', 'CommissioningScheduledDate',
        'CountryNetSuiteID', 'NewWaterSourceToBeDeveloped', 'WaterFeeStructure',
        'AverageHouseholdIncome', 'AverageHouseholdIncomeUnits'
    ]
    dt = dt.drop(rm_cols, axis=1)

    idx_cols = [
        'AssessmentID', 'AssessmentName', 'Feasible', 'Priority', 'ProjectClassification',
        'ProjectManager', 'ProjectSubclassification', 'ProjectType',
        'InstallationDate', 'CommissioningDate', 'OperatorCompensationMethod',
        'Continent', 'Country', 'Region'

    ]
    dt = dt.set_index(idx_cols)

    if rm_na:
        dt = dt[dt.apply(lambda x: x.isnull().sum() == 0, axis=1)]

    return dt