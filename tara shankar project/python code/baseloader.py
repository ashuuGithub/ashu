import os
import sys
import traceback
import logging
import csv
import pyodbc
from pprint import pprint
from datetime import datetime
import argparse

# Fetch the Global Variables
from Globals import *
from LogDbHandler import *
from Utils import *

class BaseLoader:
    UNMATCHED_STATS = 'unmatched_stats'
    UNMATCHED = 'unmatched'
    STATS = 'stats'

    def __init__(self, name, log: logging.Logger, startDate, endDate) -> None:
        self.log = log
        self.db_conn = db_conn
        self.sql_server = sql_server
        self.sql_working_database = sql_working_database
        self.sql_working_username = sql_working_username
        self.sql_working_password = sql_working_password
        self.trim_date_field = "transaction_date"
        self.sql_batch_size = sql_batch_size
        self.startDate = startDate
        self.endDate = endDate
        self.name = name
        
        self.matching_tables_to_clean = {
            self.UNMATCHED_STATS: [],
            self.UNMATCHED: [],
            self.STATS: []
        }

    def load(self):
        raise NotImplementedError(f"Loader {self.name} has not implemented the load method")

    def trim(self):
        conn = self.db_conn(self.sql_server, self.sql_working_database, self.sql_working_username, self.sql_working_password)
        cursor = conn.cursor()
        try:
            self.log.info(f'Trimming transactions on TRUST.{self.name} on or after: {self.startDate}')
            cursor.execute(f'DELETE FROM TRUST.{self.name} WHERE {self.trim_date_field} >= ?', [self.startDate])
            conn.commit()
            self.log.info(f"Completed trimming transactions on TRUST.{self.name} on or after: {self.startDate}")
        finally:
            cursor.close()
            conn.close()

    def clean_matching_tables(self):
        tables_to_clean = self.matching_tables_to_clean
        if len(tables_to_clean) > 0:
            conn = self.db_conn(self.sql_server, self.sql_working_database, self.sql_working_username, self.sql_working_password)
            cursor = conn.cursor()
            try:
                for table_to_clean in tables_to_clean:
                    cursor.execute(f"TRUNCATE TABLE TRUST.{table_to_clean}")
                conn.commit()
            finally:
                cursor.close()
                conn.close()

    def match(self, matchDate, notIncluded):
        matchers = self.get_matchers(matchDate)  # Getting the matchers for this loader
        if len(matchers) > 0:
            matchConn = self.db_conn(self.sql_server, self.sql_working_database, self.sql_working_username, self.sql_working_password)
            matchCursor = matchConn.cursor()
            try:
                for matcher in matchers:
                    self.log.info(f"Started identifying Unmatched transactions ({matcher}) on TRUST tables on or after: {matchDate} up to and not including: {notIncluded}")
                    matchCursor.execute(matchers[matcher]['sql'], matchers[matcher]['parameters'])
                    self.log.info(f'Finished identifying Unmatched transactions ({matcher}) on TRUST tables on or after: {matchDate} up to and not including: {notIncluded}')
                matchConn.commit()
            finally:
                matchCursor.close()
                matchConn.close()

    def get_matchers(self, match_date):
        # Some loaders don't have matchers
        return {}
