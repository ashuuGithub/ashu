import os
import tempfile
import logging
import pyodbc
import boto3
import argparse
import pytz
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from LogDbHandler import *
from Globals import *

# Define date formats
format_yyyy_mm_dd = "%Y-%m-%d"
format_mm_dd_yyyy = "%m-%d-%Y"

# Global variables
session = None
logging_init_count = 0
log_conn = None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_s3_session():
    global session
    if session is None:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    return session

def open_s3_file(file_name):
    # Create an S3 client session
    session = get_s3_session()
    s3_client = session.client('s3')

    # Retrieve the file object from S3
    response = s3_client.get_object(Bucket=aws_bucket_name, Key=file_name)
    print("Open S3 file:", file_name, "Response:", response)

    # Read the file content line by line
    file_content = response['Body'].iter_lines()
    return file_content

def s3_download(s3_client, file_key):
    f = tempfile.mkstemp(suffix='.tmp')
    tmpPath = f[1]
    os.close(f[0])
    s3_client.download_file(aws_bucket_name, file_key, tmpPath)
    return tmpPath

def load_from_s3_response(response, file_object_check, process_file, startDate, endDate, s3_client):
    foundCount = response['KeyCount']
    if foundCount > 0:
        file_objects = response['Contents']
        for file_object in file_objects:
            doProcess, fileDate = file_object_check(file_object, startDate, endDate, s3_client)
            if doProcess:
                # Get the key (file path) of the object
                file_key = file_object['Key']
                log.info("Started File: " + file_key + " for date: " + str(fileDate))

                # Copy file from S3 to tmp file
                tmpPath = s3_download(s3_client, file_key)
                process_file(tmpPath, file_key, fileDate, startDate)

                # Delete tmp file
                os.remove(tmpPath)
    else:
        log.warning("No files found in: " + response['Prefix'])

def load_from_s3(fileFolder, file_object_check, process_file, startDate, endDate):
    log.info("Using S3 bucket: " + fileFolder)
    session = get_s3_session()
    s3_client = session.client('s3')
    response = s3_client.list_objects_v2(Bucket=aws_bucket_name, Prefix=fileFolder)

    while True:
        load_from_s3_response(response, file_object_check, process_file, startDate, endDate, s3_client)
        
        # S3 returns IsTruncated == True whenever there are more than 1000 file objects left
        isTruncated = response['IsTruncated']
        if not isTruncated:
            break
        continuationToken = response['NextContinuationToken']
        response = s3_client.list_objects_v2(Bucket=aws_bucket_name, Prefix=fileFolder, ContinuationToken=continuationToken)

def load_from_directory(fileFolder, dir_entry_check, process_file, startDate, endDate):
    fileDir = os.path.join(data_input_folder, fileFolder)
    log.info("Using file directory: " + fileDir)
    
    for dirEntry in os.scandir(fileDir):
        doProcess, fileDate = dir_entry_check(dirEntry, startDate, endDate)
        if doProcess:
            log.info("Started File: " + dirEntry.path + " for date: " + str(fileDate))
            process_file(dirEntry.path, dirEntry.path, fileDate, startDate)

def db_conn(sql_server, sql_database, sql_username, sql_password, use_sql_trusted_connection=sql_trusted_connection_enabled):
    if use_sql_trusted_connection:
        return pyodbc.connect("Driver={" + sql_driver + "};"
                              "Server=" + sql_server + ";"
                              "Database=" + sql_database + ";"
                              "Trusted_Connection=yes;"
                              "TrustServerCertificate=yes;"
                              "ConnectRetryCount=20;"
                              "ConnectRetryInterval=20;")
    else:
        return pyodbc.connect("Driver={" + sql_driver + "};"
                              "Server=" + sql_server + ";"
                              "Database=" + sql_database + ";"
                              "UID={" + sql_username + "};"
                              "PWD={" + sql_password + "};"
                              "Trusted_Connection=no;"
                              "TrustServerCertificate=yes;"
                              "ConnectRetryCount=20;"
                              "ConnectRetryInterval=20;")

def shiftDates(startDate, endDate, offset):
    tempOneDayBehind = datetime.strptime(startDate, "%Y-%m-%d") + timedelta(days=-offset)
    startDate = datetime.strftime(tempOneDayBehind, "%Y-%m-%d")
    tempOneDayBehind = datetime.strptime(endDate, "%Y-%m-%d") + timedelta(days=-offset)
    endDate = datetime.strftime(tempOneDayBehind, "%Y-%m-%d")
    return startDate, endDate

def filter_file_by_modified_time(dirEntry, startDate, endDate):
    fileStartDateTime = datetime.strptime(startDate, format_yyyy_mm_dd) + timedelta(days=+1)
    fileEndDateTime = datetime.strptime(endDate, format_yyyy_mm_dd) + timedelta(days=+1)
    modified_time = os.path.getmtime(dirEntry)
    dt_m = datetime.fromtimestamp(modified_time)
    fileDate = dt_m.strftime(format_yyyy_mm_dd)
    
    if dt_m < fileStartDateTime or dt_m >= fileEndDateTime:
        return False, fileDate
    return True, fileDate

def filter_file_by_modified_time_s3(file_object, startDate, endDate, s3_client):
    fileStartDateTime = datetime.strptime(startDate, format_yyyy_mm_dd) + timedelta(days=+1)
    fileEndDateTime = datetime.strptime(endDate, format_yyyy_mm_dd) + timedelta(days=+1)

    # Make datetime values timezone-aware to match the LastModified value from S3
    fileStartDateTime = fileStartDateTime.replace(tzinfo=pytz.utc)
    fileEndDateTime = fileEndDateTime.replace(tzinfo=pytz.utc)

    # Use Metadata file-date value if available
    response = s3_client.head_object(Bucket=aws_bucket_name, Key=file_object["Key"])
    metadata = response['Metadata']
    
    if 'file-date' in metadata:
        fileDate = metadata['file-date']
        dt_m = datetime.strptime(fileDate, format_yyyy_mm_dd).replace(tzinfo=pytz.utc)
    else:
        dt_m = file_object["LastModified"]
        fileDate = dt_m.strftime(format_yyyy_mm_dd)
    
    if dt_m < fileStartDateTime or dt_m >= fileEndDateTime:
        return False, fileDate
    return True, fileDate

def filter_file_by_filename_date(dirEntry, startDate, endDate, with_dashes=True, format='ymd'):
    return filter_file_by_filename_date_common(dirEntry.path, startDate, endDate, with_dashes, format)

def filter_file_by_filename_date_s3(file_object, startDate, endDate, with_dashes=True, format='ymd'):
    return filter_file_by_filename_date_common(file_object["Key"], startDate, endDate, with_dashes, format)

def filter_file_by_filename_date_common(file_name, startDate, endDate, with_dashes=True, format='ymd'):
    fileStartDateTime = datetime.strptime(startDate, format_yyyy_mm_dd) + timedelta(days=+1)
    fileEndDateTime = datetime.strptime(endDate, format_yyyy_mm_dd) + timedelta(days=+1)

    fileStartDate = fileStartDateTime.strftime(format_yyyy_mm_dd)
    fileEndDate = fileEndDateTime.strftime(format_yyyy_mm_dd)

    currentDate = fileStartDate
    currentDateTime = fileStartDateTime

    while currentDate < fileEndDate:
        if format == 'ymd':
            testDate = currentDate if with_dashes else currentDate.replace("-", "")
        else:
            testDate = currentDate[5:7] + '-' + currentDate[8:10] + '-' + currentDate[0:4]
            testDate = testDate if with_dashes else testDate.replace("-", "")
        
        if testDate in file_name:
            break
        else:
            currentDateTime += timedelta(days=1)
            currentDate = datetime.strftime(currentDateTime, format_yyyy_mm_dd)
    
    if currentDate >= fileEndDate:
        return False, currentDate
    return True, currentDate

def term_logger():
    global log_conn
    global logging_init_count
    
    if logging_init_count == 0:
        return
    
    logging_init_count -= 1
    if logging_init_count == 0:
        if log_conn:
            log_conn.close()

def init_logger():
    global log_conn
    global logging_init_count
    
    # Set db handler for root logger
    if logging_init_count > 0:
        logging_init_count += 1
        return
    
    logging_init_count = 1
    
    # Set up loggers
    fileHandler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    fileHandler.setFormatter(formatter)
    fileHandler.datefmt = '%Y-%m-%d %H:%M:%S'
    fileHandler.setLevel(log_error_level)

    # Define a Handler which writes INFO messages or higher to the console
    console = logging.StreamHandler()
    console.setLevel(log_error_level)
    
    # Set a format which is simpler for console use
    console_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    console.setFormatter(console_formatter)
    
    logger = logging.getLogger('TRUST_LOGGER')
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(console)
    logger.addHandler(fileHandler)
    
    if log_to_db:
        # Make the connection to database for the logger
        log_conn = db_conn(sql_server, sql_working_database, sql_working_username, sql_working_password)
        log_cursor = log_conn.cursor()
        logdb = LogDbHandler(log_conn, log_cursor, db_tbl_log)
        logdb.name = 'logdb'
        logger.addHandler(logdb)

def files_available_check(startDate, endDate, dirEntry, log):
    """Determine if files are available to load."""
    today = datetime.today().strftime('%Y-%m-%d')
    if startDate >= today:
        log.warning('Start date supplied is today or greater. Start date must be in the past.')
        return 1
    if endDate > today:
        log.warning('End date supplied is later than today. Setting End Date to today and continuing.')
        endDate = today
    
    files = ['AmericanExpress']
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2022-01-01', end='2099-12-31').to_pydatetime()
    dt = datetime.combine(datetime.today(), datetime.min.time())
    
    if dt not in holidays and dt.weekday() < 5:
        files.append('BAT')
        files.append('Cybersource')
    
    # Test for presence of EMAF transactions loaded to DATADB
    connDataDb = db_conn(sql_datastore_server, sql_datastore_database, sql_datastore_username, sql_datastore_password)
    cursorDataDb = connDataDb.cursor()
    cursorDataDb.execute('SELECT COUNT(*) FROM EMAF.CREDIT_RECN_DETAIL (NOLOCK) WHERE created > ?', [startDate])
    emafCount = cursorDataDb.fetchone()[0]

    if emafCount < 10000:
        files.append('EMAF')
    else:
        log.debug("EMAF records present for start date. Records present: " + str(emafCount))
    
    cursorDataDb.close()
    connDataDb.close()
    
    files.append('GL')
    files.append('PayPal')
    files.append('Shift4')
    files.append('Telecheck')

    nowTime = datetime.strptime(startDate, '%Y-%m-%d') + timedelta(days=1)
    now = nowTime.strftime('%Y-%m-%d')
    log.info('Evaluating files to import for TRUST on: ' + now)

    alreadyCheckedForLastCybersourceFile = False
    for fname in os.listdir(dirEntry):
        modtime = datetime.fromtimestamp(os.stat(os.path.join(dirEntry, fname)).st_mtime)
        out = modtime.strftime("%Y-%m-%d")
        
        if not alreadyCheckedForLastCybersourceFile and os.path.exists(os.path.join(dirEntry, f'Cybersource/TransactionDetailReport_Daily_Classic_stjude_dh_wichita.{out}.xml')):
            files.remove('Cybersource')
            alreadyCheckedForLastCybersourceFile = True
        
        if out == now:
            continue
        else:
            try:
                files.remove(fname)
            except ValueError as e:
                continue
            
            log.debug(fname + " has changed today at: " + modtime.strftime('%Y-%m-%d %H:%M'))
    
    if len(files) > 0:
        log.info("Awaiting: " + str(files) + " to arrive.")
        return 1  # Keep waiting

    log.info('Files are available to successfully process TRUST loads.')
    return 0  # Success - all files available
