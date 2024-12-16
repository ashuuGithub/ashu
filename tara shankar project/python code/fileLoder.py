import os
import sys
import traceback
import csv
import logging
import argparse
from datetime import datetime
from enum import Enum
from pprint import pprint

import pyodbc
from Globals import *
from LogDbHandler import *
from Utils import *
from BaseLoader import BaseLoader

class FilterBy(Enum):
    MODIFIED_TIME = 1
    FILENAME_DATE = 2

class FileLoader(BaseLoader):
    def __init__(self, name, log: logging.Logger, startDate, endDate) -> None:
        super().__init__(name, log, startDate, endDate)
        self.file_folder = None
        self.can_use_s3 = False
        self.filter_by = FilterBy.FILENAME_DATE
        self.filename_has_dashes = True
        self.filename_date_format = "ymd"

    def file_object_check(self, file_object, startDate, endDate, s3_client):
        (startDate, endDate) = self.transform_dates(startDate, endDate)
        if self.filter_by == FilterBy.FILENAME_DATE:
            (filtered, fileModifiedTime) = filter_file_by_filename_date_s3(
                file_object, startDate, endDate, self.filename_has_dashes, self.filename_date_format)
        else:
            (filtered, fileModifiedTime) = filter_file_by_modified_time_s3(
                file_object, startDate, endDate, s3_client)
        
        if not filtered or self.filter_out_file_name(file_object):
            return False, fileModifiedTime
        return True, fileModifiedTime

    def dir_entry_check(self, dirEntry, startDate, endDate):
        # Only process files between the startDate and endDate using the embedded date in the filename
        (startDate, endDate) = self.transform_dates(startDate, endDate)
        if self.filter_by == FilterBy.FILENAME_DATE:
            (filtered, fileDate) = filter_file_by_filename_date(
                dirEntry, startDate, endDate, self.filename_has_dashes, self.filename_date_format)
        else:
            (filtered, fileDate) = filter_file_by_modified_time(
                dirEntry, startDate, endDate)
        
        if not dirEntry.is_file() or not filtered or self.filter_out_file_name(dirEntry.path):
            return False, fileDate
        return True, fileDate

    def filter_out_file_name(self, file_path) -> bool:
        return False

    def dir_entry_custom_check(self, path) -> bool:
        return self.filter_out_file_name(path)

    def file_object_custom_check(self, file_object) -> bool:
        return self.filter_out_file_name(file_object["Key"])

    def process_file(self, file_path, file_name, fileDate, startDate):
        raise NotImplementedError(f"Loader {self.name} has not implemented the process file method")

    def transform_dates(self, startDate, endDate):
        return startDate, endDate

    def load(self):
        self.log.info(f"Started {self.name} from files for {self.startDate} to but not including {self.endDate}")
        if self.can_use_s3 and use_s3_buckets_enabled:
            load_from_s3(self.file_folder, self.file_object_check, self.process_file, self.startDate, self.endDate)
        else:
            load_from_directory(self.file_folder, self.dir_entry_check, self.process_file, self.startDate, self.endDate)
        self.log.info(f"Finished {self.name} Load")
