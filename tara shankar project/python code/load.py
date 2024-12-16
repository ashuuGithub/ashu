import os
import sys
import importlib
import argparse
from datetime import datetime
import time

# Fetch the Global Variables
from Globals import *
from Utils import *
import JobExecHistory
import GiftMatch
import CollectStats
import ReceiptLedgerUnresolved
import DiscrepanciesFromXLS
from loaders import BAI, ACH, Wires, EFT, Amazon, AMEX, APG, ApplePay, Benevity, CardPayment, ChargeProcessing, Cybersource, EMAF, EPP, GooglePay, IPay, Metavante, Paypal, SHIFT4, SHIFT4_ACH, Telecheck, VSD, ReceiptLedger, BAIEnrichment, TriangleMatch

# We need to import all the loaders so we can dynamically call them
TRUSTED_LOADERS = {
    "BAI": BAI,
    "ACH": ACH,
    "Wires": Wires,
    "EFT": EFT,
    "Amazon": Amazon,
    "AMEX": AMEX,
    "APG": APG,
    "ApplePay": ApplePay,
    "Benevity": Benevity,
    "CardPayment": CardPayment,
    "ChargeProcessing": ChargeProcessing,
    "Cybersource": Cybersource,
    "EMAF": EMAF,
    "EPP": EPP,
    "GooglePay": GooglePay,
    "IPay": IPay,
    "Metavante": Metavante,
    "Paypal": Paypal,
    "SHIFT4": SHIFT4,
    "SHIFT4_ACH": SHIFT4_ACH,
    "Telecheck": Telecheck,
    "VSD": VSD,
    "ReceiptLedger": ReceiptLedger,
    "BAIEnrichment": BAIEnrichment,
    "TriangleMatch": TriangleMatch
}

def get_sanitized_loader(input_loader):
    sanitized_input_loader = input_loader.strip()
    if sanitized_input_loader not in TRUSTED_LOADERS:
        raise ValueError(f"Unauthorized loader: {sanitized_input_loader}.")
    return sanitized_input_loader

def str2bool(value):
    return value.lower() in ['true', '1', 't', 'y', 'yes']

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--loader", type=str, help='Specify the Loader Name')
parser.add_argument("-s", "--startDate", type=str, help='Specify Start Date to include YYYY-MM-DD')
parser.add_argument("-e", "--endDate", type=str, help='Specify End Date up to but not including YYYY-MM-DD')
parser.add_argument("-t", "--trim", type=str2bool, nargs='?', const=True, default=True, help="Specify True or False to have the loaders trim their tables or not.")
parser.add_argument("-a", "--addRecords", type=str2bool, nargs='?', const=True, default=True, help="Specify True or False to have the loaders add records or not.")
args = parser.parse_args()

startDate = None
endDate = None
jobExecID = None

if args.startDate and args.endDate:
    try:
        datetime.strptime(args.startDate, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid start date: {args.startDate!r}")
    
    try:
        datetime.strptime(args.endDate, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid end date: {args.endDate!r}")
    
    startDate = args.startDate
    endDate = args.endDate
elif args.startDate is None and args.endDate is None:
    log.info('Getting Start and End Date from Job_Exec_History table')
    tmpStartDate, tmpEndDate, jobExecID = JobExecHistory.start_next_execution_from_db()
    currentDate = datetime.today().strftime('%Y-%m-%d')
    startDate = str(tmpStartDate)
    
    if datetime.strptime(startDate, "%Y-%m-%d") >= datetime.strptime(currentDate, "%Y-%m-%d"):
        log.info('Start date in the future, process exiting.')
        JobExecHistory.delete_current_execution_db(jobExecID)
        term_logger()
        exit(0)
    
    endDate = str(tmpEndDate)

if startDate is None and endDate is None:
    log.warning('Parameter Start Date not specified, but End Date was specified. Exiting')
    exit(0)
if startDate is not None and endDate is None:
    log.warning('Parameter End Date not specified, but Start Date was specified. Exiting')
    exit(0)

# Initialize logger
init_logger()

if jobExecID:
    log.info(f'Job Exec ID: {str(jobExecID)}')

# Start run
log.info(f'Initiating TRUST loadAll for startDate: {startDate} to but not including: {endDate}')

# Only check for files being available if running a full load without date overrides
if args.loader is None and args.startDate is None and args.endDate is None:
    while filesAvailableCheck(startDate, endDate, data_input_folder, log) != 1:
        time.sleep(60)

execute_match_and_stats = False

if args.loader:
    sanitized_loader = get_sanitized_loader(args.loader)
    if sanitized_loader not in TRUSTED_LOADERS:
        raise ValueError(f"Loader {sanitized_loader} is not trusted")
    else:
        for loader in TRUSTED_LOADERS:
            if f"loaders.{sanitized_loader}" == f"loaders.{loader}":
                sanitized_loader = f"loaders.{loader}"
                break
        class_loader = TRUSTED_LOADERS[sanitized_loader]
        loaders[sanitized_loader] = class_loader(sanitized_loader, log, startDate, endDate)
        execute_match_and_stats = True
else:
    loader_files = []
    loader_priority = []
    try:
        with open('./loaders/priority.txt') as loader_priority_file:
            loader_priority = [line.rstrip("\n") for line in loader_priority_file.readlines()]
    except:
        log.info('Could not read loader priority file.')

    loader_files = []
    for f in os.listdir("./loaders/"):
        loaderName, extension = os.path.splitext(f)
        if loaderName == "_pycache_" or loaderName == "priority":
            continue
        order = 100
        try:
            order = loader_priority.index(loaderName)
            loader_files.append({
                'loader': next(filter(lambda x: x == loaderName, TRUSTED_LOADERS)),
                'order': order
            })
        except ValueError:
            log.info(f'Loader file "{loaderName}" wasn\'t found in the priority.txt file. This loader won\'t be processed.')
    
    loader_files.sort(key=lambda x: x['order'])

    for sorted_loader in loader_files:
        loaderName = sorted_loader['loader']
        module = importlib.import_module(f"loaders.{loaderName}")
        class_loader = getattr(module, loaderName)
        loaders[loaderName] = class_loader(loaderName, log, startDate, endDate)

# Execute loaders
for loader in loaders:
    log.info(f">>>>Starting the {loader} loader<<<<<<")
    if args.trim:
        loaders[loader].trim()
    if args.addRecords:
        loaders[loader].load()
    log.info(f">>>>Finishing the {loader} loader<<<<")

if execute_match_and_stats:
    DiscrepanciesFromXLS.load()
    GiftMatch.load(loaders, endDate)
    CollectStats.collect(loaders, startDate, endDate, log)
    ReceiptLedgerUnresolved.load()

# Update Job History
if jobExecID:
    JobExecHistory.end_current_execution_db(jobExecID, 'Success')
    log.info(f'Completed TRUST loadAll for startDate: {startDate} to but not including: {endDate}')

except pyodbc.OperationalError as e:
    log.error(f"Error on line {sys.exc_info()[-1].tb_lineno}: {repr(e)}")
    log.error(str(traceback.format_exc().splitlines())[0:2000])
except Exception as e:
    log.error(f"Error on line {sys.exc_info()[-1].tb_lineno}: {repr(e)}")
    log.error(str(traceback.format_exc().splitlines())[0:2000])
finally:
    term_logger()
