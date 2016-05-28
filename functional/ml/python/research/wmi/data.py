""" Water Missions International Data Manipulation

This module contains methods for retrieving and storing WMI datasets
"""

# Authors: Eric Czech

import os
import re
import pandas as pd
import numpy as np
from research.wmi import transforms
from IPython.core.debugger import Tracer


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
        'Region', 'ProjectClassification', 'ProjectSubclassification',
        'ProjectType', 'Priority'
    ]
    for c in c_category:
        d[c] = d[c].where(d[c].notnull(), 'Unknown')

    c_date = ['AssessmentCompletedOn']
    for c in c_date:
        d[c] = pd.to_datetime(d[c])

    c_non_null = ['AssessmentID']
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


def get_project_timeline():
    d = load_table('SafeWaterProjectMonthlySummary')

    # Remove invalid dates
    d = d[d['MonthAndYear'] != '2004-10-28']
    d = d[d['MonthAndYear'] != '2018-07-28']

    # Adjust for apparent date typo
    d['MonthAndYear'] = d['MonthAndYear'].where(d['MonthAndYear'] != '2105-06-28', '2015-06-28')
    d['MonthAndYear'] = d['MonthAndYear'].apply(lambda x: pd.to_datetime(x[:7]))

    # Rename date column
    d = d.rename(columns={'MonthAndYear': 'Date'})

    # Remove "Additional*" fields
    d = d[[c for c in d if not c.startswith('Additional')]]

    # Assessment ID + date conflicts do occur, so choose arbitrarily from between them
    d = d.groupby(['AssessmentID', 'Date'], group_keys=False).apply(lambda x: x.head(1))

    d['AssessmentID'] = pd.to_numeric(d['AssessmentID'], errors='coerce')

    d = d[d['AssessmentID'].notnull()]
    d = d[d['Date'].notnull()]

    d = d.set_index(['Date', 'AssessmentID'])
    for c in d:
        d[c] = pd.to_numeric(d[c], errors='coerce')

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


def convert_water_quality_units(g):

    def validate_units(g, expected, unit_type):
        unit_unique = g['Units'].unique()
        rem = np.setdiff1d(unit_unique, expected)
        assert len(rem) == 0, \
            'The following {} units are not yet supported: {}'.format(unit_type, rem)

    # Convert known concentration units
    if 'mg/L' in g['Units'].values or 'ppb' in g['Units'].values:
        validate_units(g, ['mg/L', 'ppb'], 'concentration')

        def get_value(r):
            r['Value'] = float(r['Value'])
            if r['Units'] == 'mg/L':
                return r['Value']
            elif r['Units'] == 'ppb':
                return r['Value'] / 1000.
            else:
                return None
        g['Value'] = g[['Units', 'Value']].apply(get_value, axis=1)
        assert g['Value'].isnull().sum() == 0, 'Found null value for converted units'

        # Assume everything has been converted to mg/L
        g['Units'] = 'mg/L'

    # Convert known temp units
    elif 'Celsius' in g['Units'].values or 'Fahrenheit' in g['Units'].values:
        validate_units(g, ['Celsius', 'Fahrenheit'], 'temperature')

        def get_value(r):
            r['Value'] = float(r['Value'])
            if r['Units'] == 'Celsius':
                return r['Value']
            elif r['Units'] == 'Fahrenheit':
                return (r['Value'] - 32.) * (5/9.)
            else:
                return None
        g['Value'] = g[['Units', 'Value']].apply(get_value, axis=1)
        assert g['Value'].isnull().sum() == 0, 'Found null value for converted units'

        # Assume everything has been converted to Celsius
        g['Units'] = 'Celsius'

    # Convert known coliform unit values
    elif 'CFU/100ml' in g['Units'].values:
        validate_units(g, ['CFU/100ml'], 'coliform')

        # Convert "TNTC" (too numerous to count -- DNPC is spanish for the same)
        # to sentinel max value
        SENTINEL_MAX_VAL = 5000.
        tntc_cols = ['too numerous to count ', 'tntc', 'dnpc', 'tnct', '"dnpc"', 'ntc', 'tmtc', 'ttc', 'tnt']
        g['Value'] = np.where(g['Value'].str.lower().isin(tntc_cols).values, SENTINEL_MAX_VAL, g['Value'].values)

        g['Value'] = pd.to_numeric(g['Value'], errors='coerce')
        g = g[g['Value'].notnull()]

        # Cap values at sentinel max
        g['Value'] = g['Value'].apply(lambda x: x if x < SENTINEL_MAX_VAL else SENTINEL_MAX_VAL)

    # Otherwise, verify that only one unit scale is present and do nothing else
    else:
        unit_unique = g['Units'].unique()
        assert len(unit_unique) == 1, \
            'Found multiple units for the same parameter (units = {})'.format(unit_unique)

    # Finally, ensure all measurements can be represented as floating point
    g['Value'] = g['Value'].astype(np.float64)
    return g


def get_water_quality_data():

    # Load water quality test data
    d_main = load_table('WaterQualityTests')
    d_main['TestID'] = d_main['TestID'].astype(np.int64)
    d_main['AssessmentID'] = d_main['AssessmentID'].astype(np.int64)
    d_main['DistributionPointID'] = d_main['DistributionPointID'].astype(np.int64)
    d_main = d_main.drop([c for c in d_main if c.startswith('GPS')], axis=1)

    # Load water quality measurements (over time)
    d_time = load_table('WaterQualityTestParameters')
    d_time['TestID'] = d_time['TestID'].astype(np.int64)
    d_time = d_time.drop('AnalysisMethod', axis=1)

    # Load distribution point metadata
    d_dist = load_table('DistributionPoints')
    d_dist['AssessmentID'] = d_dist['AssessmentID'].astype(np.int64)
    d_dist['DistributionPointID'] = d_dist['DistributionPointID'].astype(np.int64)
    d_dist = d_dist.drop('WaterSourceTypeOther', axis=1)

    # Merge all of the above
    d = pd.merge(d_main, d_time, on='TestID', how='inner')
    d = pd.merge(d, d_dist, on=['DistributionPointID', 'AssessmentID'], how='inner')

    # Remove records with unknown units or unknown value
    d = d[d['Units'].notnull()]
    d = d[d['Value'].notnull()]

    # Remove measurements with this parameter and unit combination (not sure how to convert them)
    mask = (d['Parameter'] == 'Chlorine, Free') & (d['Units'] == 'CFU/100ml')
    d = d[~mask.values]

    # Make parameter name all ucfirst
    d['Parameter'] = d['Parameter'].str.title()

    assert d['AssessmentID'].isnull().sum() == 0, 'Found null assessment ID in water quality measurements'
    assert d['SampleDateAndTime'].isnull().sum() == 0, 'Found null sample date in water quality measurements'

    # Convert date fields
    def to_monthly_date(x):
        return pd.to_datetime(x.apply(lambda x: None if pd.isnull(x) else x[:7]))

    d['Date'] = to_monthly_date(d['SampleDateAndTime'])
    d = d.drop('SampleDateAndTime', axis=1)
    d['AnalysisDateAndTime'] = to_monthly_date(d['AnalysisDateAndTime'])
    d['WaterMeterReplacementDate'] = to_monthly_date(d['WaterMeterReplacementDate'])

    # Convert measurements to common units
    d = d.groupby('Parameter', group_keys=False).apply(convert_water_quality_units)
    assert d.groupby('Parameter').apply(lambda x: len(x['Units'].unique())).max() == 1, \
        'Found parameter with multiple unit scales'

    # Remove columns with all null values
    d = d[[c for c in d if d[c].notnull().sum() > 0]]

    return d


def resolve_duplicate_water_quality_values(d_qual):
    uniq_wq_cols = ['AssessmentID', 'DistributionPointID', 'Parameter', 'Date']

    def resolve_multiple_measurements(g):
        """ Take the median value when the same measurement is taken multiple times in a month

        To give a sense of how often this is necessary, at TOW:
            1 measurement: 93%
            2 measurements: 5%
            3 measurements: 1%
            4+ measurements: 1%
        """
        if len(g) == 1:
            return g
        # Take middle-valued or measurement, or the one before it (when length of group is even)
        return g.sort_values('Value').iloc[int(len(g)/2.)]

    # Group data by desired level of uniqueness and split into two groups of groups, one with
    # multiple measurements and one without
    grp = d_qual.groupby(uniq_wq_cols)
    d1 = grp.filter(lambda x: len(x) == 1)  # Group with no duplicates
    d2 = grp.filter(lambda x: len(x) > 1) # Group with duplicates
    d2 = d2.groupby(uniq_wq_cols, group_keys=False).apply(resolve_multiple_measurements)

    # Combine resolved measurements with unique ones
    d_resolved = d1.append(d2)

    assert len(d_qual.groupby(uniq_wq_cols)) == len(d_resolved), \
        'Expected number of groups not equal to actual after duplicate measurement value resolution'

    return d_resolved


def resolve_water_quality_metadata(d):
    uniq_wq_cols = ['AssessmentID', 'DistributionPointID', 'Date']
    ts_cols = ['Parameter', 'Value']
    meta_cols = np.setdiff1d(d.columns.tolist(), ts_cols)

    def find_most_frequent(g):
        r = g[ts_cols]
        # Fetch the most frequently occurring metadata value across all parameters for a single date
        for c in meta_cols:
            if g[c].notnull().sum() == 0:
                r[c] = g[c].iloc[0]
            else:
                r[c] = g[c].value_counts().idxmax()
        return r
    return d.groupby(uniq_wq_cols, group_keys=False).apply(find_most_frequent)


def pivot_water_quality_measurements(d):
    cols = d.columns.tolist()
    meta_cols = list(np.setdiff1d(cols, ['Parameter', 'Value', 'Date']))
    d_ts = d.set_index(meta_cols + ['Date', 'Parameter'])['Value'].unstack()
    d_ts.columns = ['WQ_{}'.format(c) for c in d_ts]

    # idx_cols = list(d_ts.index.names)
    d_ts = d_ts.reset_index()

    max_freq = d_ts.groupby(['AssessmentID', 'DistributionPointID', 'Date']).size().max()
    if max_freq > 1:
        Tracer()()
    assert max_freq == 1, 'Found multiple records for the same distribution point and date'

    # from IPython.core.debugger import Tracer
    # def interpolate(g):
    #     Tracer()()
    #     return g.set_index(idx_cols).sort_index(level=-1).interpolate()
    # d_ts = d_ts.groupby(['AssessmentID', 'DistributionPointID', 'Date'], group_keys=False).apply(interpolate)

    return d_ts

# ----- Data Views ----- #


def get_project_view_01(rm_na=False, rm_test_projects=True):
    """ Raw project meta data joined to raw timeline data
    :param rm_na: Remove rows with all nulls
    :param rm_test_projects: Remove assessments/projects used for testing
    """
    # Start with project summary data
    d_proj = get_project_summary()

    # Add country metadata
    d_cty = get_country_data()
    d_proj = pd.merge(d_proj, d_cty, how='left', on='Country')

    # Add project details
    d_proj_det = get_project_details()
    d = pd.merge(d_proj, d_proj_det, how='left', on='AssessmentID')

    # Remove sparse or seemingly useless fields
    rm_cols = [
        'Budget', 'Population', 'TotalPeopleServedDisasterResponse', 'AnticipatedPeopleServed',
        'CountryCode3', 'AssessmentCompletedOn', 'CommissioningScheduledDate',
        'CountryNetSuiteID', 'NewWaterSourceToBeDeveloped', 'WaterFeeStructure',
        'AverageHouseholdIncome', 'AverageHouseholdIncomeUnits'
    ]
    d = d.drop(rm_cols, axis=1)

    # Move all non-numeric fields to index
    idx_cols = [
        'AssessmentID', 'AssessmentName', 'Feasible', 'Priority', 'ProjectClassification',
        'ProjectManager', 'ProjectSubclassification', 'ProjectType',
        'InstallationDate', 'CommissioningDate', 'OperatorCompensationMethod',
        'Continent', 'Country', 'CountryCode2', 'Region'

    ]
    d = d.set_index(idx_cols)
    if rm_na:
        d = d[d.apply(lambda x: x.isnull().sum() == 0, axis=1)]

    # Attach timeline data
    d_ts = get_project_timeline()
    idx_meta = list(d.index.names)
    idx_time = list(d_ts.index.names)
    val_meta = list(d.columns)
    val_time = list(d_ts.columns)

    # Verify that all time-based data has corresponding static data (the reverse is OK)
    id1 = d_ts.reset_index()['AssessmentID'].unique()
    id2 = d.reset_index()['AssessmentID'].unique()
    assert len(np.setdiff1d(id1, id2)) == 0, 'Found assessments in timeseries data not present in static details'

    # Merge static data with time-based data
    d = pd.merge(d.reset_index(), d_ts.reset_index(), how='outer', on='AssessmentID')
    d = d.set_index(list(np.unique(idx_meta + idx_time)))

    # Remove testing/debugging projects, if specified to do so
    if rm_test_projects:
        mask = d.reset_index()['AssessmentName'].apply(lambda x: 'test project' in x.lower())
        d = d[~mask.values]

    return d, idx_meta, val_meta, idx_time, val_time


def get_prepared_assessments(d, idx_meta, val_meta, idx_time, val_time):
    def prep_assessment(g):
        if len(g) == 1:
            g['RowId'] = 1
            return g
        assert g['Date'].isnull().sum() == 0, \
            'Found unexpected null dates for assessment {}'.format(g['AssessmentID'].iloc[0])

        min_date = g['Date'].min()
        max_date = g['Date'].max()
        g = g.set_index('Date')
        g['RowId'] = np.arange(1, len(g) + 1)

        # Create date index (make sure frequency will result in date matching individual dates)
        idx = pd.date_range(min_date, max_date, freq='MS', name='Date')
        g = g.reindex(idx)

        # Add flag indicating any interpolated values
        g['Interpolated'] = g['RowId'].isnull()

        # Completely fill in missing, static fields
        cols1 = [c for c in (idx_meta + val_meta) if c != 'Date']
        g[cols1] = g[cols1].ffill().bfill()

        # Forward fill time-based values
        cols2 = [c for c in (idx_time + val_time) if c != 'Date']
        g[cols2] = g[cols2].ffill()

        g = g.reset_index()

        assert g['AssessmentID'].isnull().sum() == 0, 'Found null AssessmentID after reindexing'
        g['AssessmentID'] = g['AssessmentID'].astype(np.int64)
        assert g['Date'].isnull().sum() == 0, 'Found null Date after reindexing'
        return g

    idx_cols = list(d.index.names)
    d = d.reset_index()
    assert d['AssessmentID'].isnull().sum() == 0, 'Found null AssessmentIDs'

    d = d.groupby('AssessmentID', group_keys=False).apply(prep_assessment).drop('RowId', axis=1)
    return d.set_index(idx_cols + ['Interpolated'])


def get_project_view_02():
    """ Interpolated project timeline data with benchmarks

    Note that this view will NOT return records for projects with no data in
    ProjectsInformation (i.e. timeline data).  If you need these records, see
    get_project_view_01 instead.
    """
    d, idx_meta, val_meta, idx_time, val_time = get_project_view_01(rm_na=False)
    d = get_prepared_assessments(d, idx_meta, val_meta, idx_time, val_time)
    d = transforms.add_calculated_quantities(d)
    d = transforms.add_benchmarks(d, {'All': ['Date'], 'Country': ['Date', 'Country']})
    return d, idx_meta, val_meta, idx_time, val_time


def get_project_view_03(rm_test_projects=True):
    """ Project water quality:
    1. Includes measurements over time for same distribution point
    2. No benchmarks
    """

    # Load project timeline data (for now ignore everything except project metadata)
    d_proj, idx_meta, val_meta, idx_time, val_time = get_project_view_01(rm_na=False, rm_test_projects=rm_test_projects)

    d_proj = d_proj.reset_index()[idx_meta].drop_duplicates()
    idx_proj = list(d_proj.columns)
    d_proj['AssessmentID'] = d_proj['AssessmentID'].astype(np.int64)

    d_qual = get_water_quality_data()
    d_qual = resolve_duplicate_water_quality_values(d_qual).drop('Units', axis=1)

    d = pd.merge(d_qual, d_proj, on='AssessmentID', how='inner')

    # Find most frequently occurring metadata values for the same project + date
    # combination (note that this is possible because of multiple "Parameter" settings per date)
    d = resolve_water_quality_metadata(d)

    # Pivot water quality metrics into columns
    d = pivot_water_quality_measurements(d)

    # Add identifier for each distribution point
    def get_identifier(r):
        return '{}:{} ({} | {})'.format(
            r['AssessmentID'], r['DistributionPointID'],
            r['AssessmentName'], r['DistributionPointName']
        )
    d['DistributionPointIdentifier'] = d.apply(get_identifier, axis=1)

    idx_qual = np.setdiff1d(d.columns.tolist(), idx_proj)
    return d, idx_proj, idx_qual


def get_project_view_04(add_log_transforms=True):
    """ Project water quality:
    1. Includes only the most recent measurement for each distribution point
    2. Includes benchmarks by country and overall
    """
    d, idx_proj, idx_qual = get_project_view_03(rm_test_projects=True)

    # Select most recent measurement for each distribution point
    d = d.groupby('DistributionPointIdentifier', group_keys=False)\
        .apply(lambda x: x.sort_values('Date', ascending=False).head(1))

    if add_log_transforms:
        d = transforms.log_transform_water_quality_measurements(d)

    # Add overall and country level benchmarks for each quality measurement
    d['Organization'] = 'WMI'
    idx_cols = [c for c in d if not c.startswith('WQ')]
    n_before = len(d)
    d = transforms.add_benchmarks(d.set_index(idx_cols), {'All': ['Organization'], 'Country': ['Country']})
    assert n_before == len(d), \
        'Expected {} rows after adding benchmarks but found {} instead'.format(n_before, len(d))

    return d, idx_proj, idx_qual


