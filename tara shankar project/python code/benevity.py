import os
import openpyxl
import pandas as pd
from FileLoader import FileLoader, FilterBy
import logging
from datetime import datetime, timedelta
from FixedWidthTextParser.Parser import Parser

class Benevity(FileLoader):
    headerParser = None
    transactionParser = None
    trailerParser = None
    transaction = None

    def __init__(self, name, log: logging.Logger, startDate, endDate) -> None:
        # Always call the parent class constructor first
        super().__init__(name, log, '1900-01-01', endDate)  # Replacing startDate for Benevity

        # Setting up the loader
        self.matching_tables_to_clean = ['BENEVITY_DMS']
        self.stat_queries = {
            'UNMATCHED_STATS': """
                SELECT 'Benevity' AS SOURCE, DonationDate AS TRANSACTION_DATE, MERCHANT_ID,
                SUM(TotalDonationToBeAcknowledged) AS AMOUNT, COUNT(*) AS COUNT
                FROM TRUST.BENEVITY WITH (NOLOCK)
                GROUP BY DonationDate, MERCHANT_ID
            """,
            'UNMATCHED': """
                SELECT 'BENEVITY-DMS' AS SOURCE, TRANSACTION_DATE, MERCHANT_ID,
                '' AS TRANSACTION_TIME, '' AS REQUEST_ID, '' AS MERCHANT_REF_NBR,
                '' AS RECONCILIATION_ID, '' AS DMS_FINANCIAL_ID, NULL AS APG_ID, 
                '' AS APP_NAME
                FROM TRUST.BENEVITY_DMS WITH (NOLOCK)
            """,
            'STATS': """
                SELECT DonationDate AS TRANSACTION_DATE, 'BENEVITY' AS SOURCE,
                COUNT(*) AS CNT, SUM(TOTALDONATIONTOBEACKNOWLEDGED) AS AMOUNT
                FROM TRUST.BENEVITY WITH (NOLOCK)
                GROUP BY DonationDate
            """
        }

        self.file_folder = "Benevity_test"
        self.filter_by = FilterBy.MODIFIED_TIME
        self.can_use_s3 = True

    def trim(self):
        self.log.info("Benevity won't trim")

    def is_file_processed(self, file_name):
        conn = self.db_conn(self.sql_server, self.sql_working_database, self.sql_working_username, self.sql_working_password)
        cursor = conn.cursor()
        try:
            query = "SELECT 1 FROM TRUST.Benevity_processed_files WHERE FileName = ?"
            self.log.info("Executing SQL [Query]")
            cursor.execute(query, (file_name,))
            return cursor.fetchone() is not None
        finally:
            cursor.close()
            conn.close()

    def mark_file_as_processed(self, file_name):
        conn = self.db_conn(self.sql_server, self.sql_working_database, self.sql_working_username, self.sql_working_password)
        cursor = conn.cursor()
        try:
            query = "INSERT INTO TRUST.Benevity_processed_files (FileName) VALUES (?)"
            self.log.info("Executing SQL [Query]")
            cursor.execute(query, (file_name,))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    def process_file(self, file_path, file_name, file_date, start_date):
        # Check if the file has already been processed
        if self.is_file_processed(file_name):
            self.log.info(f"Skipping already processed file: {file_name}")
            return

        conn = self.db_conn(self.sql_server, self.sql_working_database, self.sql_working_username, self.sql_working_password)
        cursor = conn.cursor()
        cursor.fast_executemany = True

        sql = '''
            INSERT INTO TRUST.BENEVITY (
                COMPANY, PROJECT, DONATIONDATE, FIRSTNAME, LASTNAME, EMAIL, ADDRESS, CITY,
                STATECODE, ZIPCODE, ACTIVITY, COMMENT, TRANSACTIONID, DONATIONFREQUENCY, CURRENCY,
                PROJECTREMOTEID, SOURCE, REASON, TOTALDONATIONTOBEACKNOWLEDGED, MATCHAMOUNT,
                CAUSESUPPORTFEE, MERCHANT_FEE, FEECOMMENT
            ) VALUES (
                ?, ?, CAST(TRIM(?) AS DATE), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        '''

        try:
            self.log.info(f"Processing file: {file_name}")
            record_count = 0
            tuples = []
            file = openpyxl.load_workbook(file_path)

            # Find the sheet whose name starts with 'DonationReport'
            donation_report_sheet = None
            for sheet_name in file.sheetnames:
                if sheet_name.startswith('DonationReport'):
                    donation_report_sheet = file[sheet_name]
                    break

            if not donation_report_sheet:
                self.log.info("No sheet found with name starting with 'DonationReport', skipping file")
                return

            # Convert the sheet to a DataFrame for easy processing
            df = pd.DataFrame(donation_report_sheet.values)
            df.columns = df.iloc[0]  # Set the first row as column headers
            df = df.drop(index=0)  # Drop the header row

            # Select relevant columns
            df = df[['COMPANY', 'PROJECT', 'DONATIONDATE', 'FIRSTNAME', 'LASTNAME',
                      'EMAIL', 'ADDRESS', 'CITY', 'STATECODE', 'ZIPCODE', 'ACTIVITY',
                      'COMMENT', 'TRANSACTIONID', 'DONATIONFREQUENCY', 'CURRENCY',
                      'PROJECTREMOTEID', 'SOURCE', 'REASON', 'TOTALDONATIONTOBEACKNOWLEDGED',
                      'MATCHAMOUNT', 'CAUSESUPPORTFEE', 'MERCHANT_FEE', 'FEECOMMENT']]

            # Convert date column to datetime
            df["DONATIONDATE"] = pd.to_datetime(df["DONATIONDATE"]).dt.date

            # Convert DataFrame to a list of tuples
            tuples = [tuple(x) for x in df.itertuples(index=False, name=None)]

            # Insert records into the database in batches
            batch_size = self.sql_batch_size
            for i in range(0, len(tuples), batch_size):
                batch = tuples[i:i + batch_size]
                cursor.executemany(sql, batch)
                conn.commit()

            self.log.info(f"Finished processing BENEVITY FILE with {len(tuples)} records.")

            # Mark the file as processed
            self.mark_file_as_processed(file_name)

        except Exception as e:
            conn.rollback()
            self.log.error(f"BENEVITY loader: Error inserting records into database: {repr(e)}")
        finally:
            cursor.close()
            conn.close()

    def get_matchers(self, match_date):
        return {
            'Benevity->DMS': {
                'sql': """
                    INSERT INTO TRUST.BENEVITY_DMS (MERCHANT_ID, COMPANY, TRANSACTION_DATE, AMOUNT)
                    SELECT MERCHANT_ID, COMPANY, TRANSACTION_DATE, amount
                    FROM (
                        SELECT MERCHANT_ID, COMPANY, DONATIONDATE AS TRANSACTION_DATE,
                        SUM(TOTALDONATIONTOBEACKNOWLEDGED) AS amount
                        FROM TRUST.BENEVITY WITH (NOLOCK)
                        GROUP BY COMPANY, DONATIONDATE, MERCHANT_ID
                    ) B
                    WHERE B.TRANSACTION_DATE = ?
                    AND NOT EXISTS (
                        SELECT 1 FROM TRUST.DMS D WITH (NOLOCK)
                        WHERE D.PAYMENTMETHODCODE = 1
                        AND D.LASTNAME = B.COMPANY
                        AND B.TRANSACTION_DATE < D.TRANSACTION_DATE
                        AND B.amount = D.AMOUNT
                    )
                """,
                'parameters': [match_date]
            }
        }

    def filter_out_file_name(self, file_path) -> bool:
        return "Benevity" not in file_path or "Thumbs.db" in file_path or "~$" in file_path
