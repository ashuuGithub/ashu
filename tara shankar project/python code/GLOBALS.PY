import os
from os.path import expanduser, join
from collections import OrderedDict
from sys import platform
import socket
import sys
import traceback
from datetime import datetime, timedelta
import logging

class FixedWidthFieldLine(object):
    def __init__(self, fields, justify='L'):
        """
        Returns a line from a list containing tuples of field values and lengths.
        Accepts justification parameter: FixedWidthFieldLine(fields [, justify])
        fields = [(value, fieldLength) [,...]]
        """
        self.fields = fields
        if justify in ('L', 'C', 'R'):
            self.justify = justify
        else:
            self.justify = 'L'

    def __str__(self):
        result = []
        for field in self.fields:
            field_value, field_length = field
            if field_value is None:
                print("NoneType discovered")
                print(str(self.fields))
                sys.exit()
            if len(field_value) > field_length:
                print(f"Too long: {field_value} exceeds {field_length}")
            if self.justify == 'L':
                result.append(field_value.ljust(field_length))
            elif self.justify == 'R':
                result.append(field_value.rjust(field_length))
            elif self.justify == 'C':
                result.append(field_value.center(field_length))
        return ''.join(result)

class FixedWidthFieldLineElement:
    def __init__(self, value='', length=0):
        """Constructor for FixedWidthFieldLineElement"""
        self.element_content = value
        self.length = length


# Set defaults
log_error_level = 'DEBUG'
log_file_path = join(expanduser("~"), 'TRUST_JOB_LOG.txt')
log_to_db_str = 'true'
db_tbl_log = 'TRUST.JOB_LOG'

# Register TRUST_LOGGER
log = logging.getLogger('TRUST_LOGGER')
log.setLevel(log_error_level)

try:
    if platform.startswith("win"):
        sql_driver = 'ODBC Driver 17 for SQL Server'
        sql_trusted_connection = 'yes'
        data_input_folder = '\\\\alsac.local\\crm\\Reconciliation\\'
    elif platform.startswith("linux"):
        sql_driver = 'ODBC Driver 18 for SQL Server'
        sql_trusted_connection = 'no'
        data_input_folder = './data/'
    else:
        sql_driver = "unknown"
        sql_trusted_connection = 'no'
        data_input_folder = './'

    sql_server = 'DATAETLD181'
    hostname = socket.gethostname()
    if 'ALSAC' in hostname:
        sql_server = 'DATAETLD1S1'
    elif 'TRUSTD' in hostname:
        sql_server = 'DATAETLD1S1'
    elif 'TRUSTQ' in hostname:
        sql_server = 'DATAETLQ1S1'
    elif 'TRUSTP' in hostname:
        sql_server = 'DATAETLP1S4'

    sql_working_database = 'Reconciliation'
    sql_datastore_server = 'DATADB'
    sql_datastore_database = 'DATASTORE'
    sql_datastore_username = 'DATADB'
    sql_batch_size_str = '10000'
    matching_window_in_days_str = '120'
    debug_enabled_str = 'false'
    use_test_dates = 'true'
    use_s3_buckets = 'false'
    aws_bucket_name = ''
    aws_access_key_id = ''
    aws_secret_access_key = ''

    # Get environment overrides
    sql_driver = os.environ.get('SQL_DRIVER', sql_driver)
    sql_server = os.environ.get('SQL_SERVER', sql_server)
    sql_trusted_connection = os.environ.get('SQL_TRUSTED_CONNECTION', sql_trusted_connection)
    sql_working_database = os.environ.get('SQL_WORKING_DATABASE', sql_working_database)
    sql_working_username = os.environ.get('SQL_WORKING_USERNAME', sql_working_username)
    sql_working_password = os.environ.get('SQL_WORKING_PASSWORD', sql_working_password)
    sql_datastore_server = os.environ.get('SQL_DATASTORE_SERVER', sql_datastore_server)
    sql_datastore_database = os.environ.get('SQL_DATASTORE_DATABASE', sql_datastore_database)
    sql_datastore_username = os.environ.get('SQL_DATASTORE_USERNAME', sql_datastore_username)
    sql_datastore_password = os.environ.get('SQL_DATASTORE_PASSWORD', sql_datastore_password)
    sql_batch_size_str = os.environ.get('SQL_BATCH_SIZE', sql_batch_size_str)
    matching_window_in_days_str = os.environ.get('MATCHING_WINDOW_IN_DAYS', matching_window_in_days_str)
    data_input_folder = os.environ.get('DATA_INPUT_FOLDER', data_input_folder)
    aws_bucket_name = os.environ.get('AWS_BUCKET_NAME', aws_bucket_name)
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', aws_access_key_id)
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY', aws_secret_access_key)
    debug_enabled_str = os.environ.get('DEBUG_ENABLED', debug_enabled_str)
    log_to_db_str = os.environ.get('LOG_TO_DB', log_to_db_str)
    use_test_dates = os.environ.get('USE_TEST_DATES', use_test_dates)
    use_s3_buckets = os.environ.get('USE_S3_BUCKETS', use_s3_buckets)

    debug_enabled = debug_enabled_str.lower() == 'true'
    log_to_db = log_to_db_str.lower() == 'true'
    use_test_dates_enabled = use_test_dates.lower() == 'true'
    use_s3_buckets_enabled = use_s3_buckets.lower() == 'true'
    sql_trusted_connection_enabled = sql_trusted_connection.lower() == 'true'

    try:
        sql_batch_size = int(sql_batch_size_str)
    except ValueError:
        log.warn(f'Invalid SQL batch size [{sql_batch_size_str}], defaulting to 10000')
        sql_batch_size = 10000

    try:
        matching_window_in_days = int(matching_window_in_days_str)
    except ValueError:
        log.warn(f'Invalid matching window in days [{matching_window_in_days_str}], defaulting to 60')
        matching_window_in_days = 60

    # Set default dates if using test dates
    if use_test_dates_enabled:
        startDate = '2022-12-01'
        endDate = '2023-02-08'
    else:
        todayDate = datetime.now()
        yesterdayDate = todayDate - timedelta(days=1)
        startDate = yesterdayDate.strftime("%Y-%m-%d")
        endDate = todayDate.strftime("%Y-%m-%d")

    if debug_enabled:
        log.info(f"SQL Driver: {sql_driver}")
        log.info(f"SQL Server: {sql_server}")
        log.info(f"SQL Trusted Connection: {sql_trusted_connection_enabled}")
        log.info(f"SQL Working Database: {sql_working_database}")
        log.info(f"Data Input Folder: {data_input_folder}")
        log.info(f"Use S3 Buckets: {use_s3_buckets_enabled}")
        log.info(f"AWS S3 Bucket Name: {aws_bucket_name}")

except Exception as e:
    log.error(f"Error on line {sys.exc_info()[-1].tb_lineno}")
    log.error(''.join(traceback.format_exc().splitlines()[:2000]))
    log.error(repr(e))
