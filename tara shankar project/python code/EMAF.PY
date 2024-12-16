from logging import Logger
from DBLoader import DBLoader

class EMAF(DBLoader):
    def __init__(self, name, log: Logger, startDate, endDate) -> None:
        super().__init__(name, log, startDate, endDate)
        self.matching_tables_to_clean = ["APG_EMAF", "CS_EMAF"]
        self.stat_queries = {
            self.UNMATCHED_STATS: [
                "SELECT 'EMAF' AS SOURCE, TRANSACTION_DATE, MERCHANT_ACCT, SUM(AMOUNT) AS AMOUNT, COUNT(*) AS COUNT "
                "FROM TRUST.EMAF WITH (NOLOCK) GROUP BY TRANSACTION_DATE, MERCHANT_ACCT",
                "SELECT 'CS-EMAF' AS SOURCE, TRANSACTION_DATE, MERCHANT_ID, SUM(AMOUNT) AS AMOUNT, COUNT(*) AS COUNT "
                "FROM TRUST.CS_EMAF WITH (NOLOCK) GROUP BY TRANSACTION_DATE, MERCHANT_ID"
            ],
            self.UNMATCHED: [
                "SELECT 'CS-EMAF' AS SOURCE, TRANSACTION_DATE, MERCHANT_ID, AMOUNT AS AMOUNT, "
                "17 AS APP_NAME, TRANSACTION_TIME AS TRANSACTION_TIME, REQUEST_ID AS REQUEST_ID, "
                "MERCHANT_REF_NBR AS MERCHANT_REF_NBR, RECONCILIATION_ID AS RECONCILIATION_ID, "
                "'' AS DMS_FINANCIAL_ID, APG_ID AS APG_ID "
                "FROM TRUST.CS_EMAF WITH (NOLOCK)"
            ],
            self.STATS: [
                "SELECT TRANSACTION_DATE, 'EMAF' AS SOURCE, COUNT(*) AS CNT, SUM(AMOUNT) AS AMOUNT "
                "FROM TRUST.EMAF WITH (NOLOCK) GROUP BY TRANSACTION_DATE"
            ]
        }

    def load(self):
        recordCount = 0
        totalAmount = 0

        self.log.info(f"Started EMAF from DATADB for {self.startDate} to but not including {self.endDate}")
        connDataDb = self.db_conn(self.sql_datastore_server, self.sql_datastore_database, self.sql_datastore_username, self.sql_datastore_password)
        conn = self.db_conn(self.sql_server, self.sql_working_database, self.sql_working_username, self.sql_working_password)
        cursorDataDb = connDataDb.cursor()
        cursor = conn.cursor()

        try:
            cursor.fast_executemany = True
            sql = """
            INSERT INTO TRUST.EMAF (
                AMOUNT, CARD_SUFFIX, CARD_TYPE, EMAF_ID, MERCHANT_ACCT, MERCHANT_REF_NBR, 
                RECONCILIATION_ID, TERMINAL_NBR, BATCH_NBR, REGISTER_NBR, POSTED_DATE, 
                TRANSACTION_DATE, TRANSACTION_TIME, EXPIRY, BIN, TRANSACTION_TYPE_CODE
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            selectSql = """
            SELECT TRANSACTION_AMT AS AMOUNT, ACCT_NBR AS CARD_NBR, LAST4, NETWORK_ID AS CARD_TYPE, 
            ALSAC_RECORD_ID AS EMAF_ID, MERCHANT_ACCT, MERCHANT_REF_NBR, WORLD_PAY_RECN_ID AS RECONCILIATION_ID, 
            TERMINAL_NBR, BATCH_NBR, REGISTER_NBR, TRANSACTION_DATE, 
            CONVERT(DATE, CONVERT(VARCHAR(8), ALSAC_FILE_ID), 112) AS POSTED_DATE, TRAN_TM, TRAN_TYPE_CD, EXP_DT 
            FROM EMAF.CREDIT_RECN_DETAIL (NOLOCK) 
            WHERE ALSAC_FILE_ID >= ? AND ALSAC_FILE_ID < ? 
            ORDER BY TRANSACTION_DATE ASC
            """
            
            cursorDataDb.execute(selectSql, [self.startDate.replace('-', ''), self.endDate.replace('-', '')])

            date = None
            tuples = []

            for row in cursorDataDb:
                if date != str(row.TRANSACTION_DATE):
                    date = str(row.TRANSACTION_DATE)

                tuple = [
                    row.AMOUNT, row.LAST4, row.CARD_TYPE, row.EMAF_ID, row.MERCHANT_ACCT, 
                    row.MERCHANT_REF_NBR, row.RECONCILIATION_ID, row.TERMINAL_NBR, row.BATCH_NBR, 
                    row.REGISTER_NBR, row.POSTED_DATE, row.TRANSACTION_DATE, 
                    row.TRAN_TM[:2] + ':' + row.TRAN_TM[2:4] + ':00', row.EXP_DT, row.CARD_NBR[:6], row.TRAN_TYPE_CD
                ]
                tuples.append(tuple)
                recordCount += 1
                totalAmount += row.AMOUNT

                if len(tuples) >= self.sql_batch_size:
                    cursor.executemany(sql, tuples)
                    conn.commit()
                    tuples.clear()

            if len(tuples) > 0:
                cursor.executemany(sql, tuples)
                conn.commit()

            self.log.info(f"Finished EMAF Database Records: {recordCount} Amount: {totalAmount:.2f}")

        except Exception as e:
            conn.rollback()
            self.log.error(f"EMAF Loader: Error inserting records into database: {repr(e)}")

        finally:
            cursorDataDb.close()
            cursor.close()
            connDataDb.close()
            conn.close()

    def get_matchers(self, matchDate):
        return {
            'CyberSource->EMAF': {
                'sql': """
                INSERT INTO TRUST.CS_EMAF (
                    TRANSACTION_DATE, REQUEST_ID, MERCHANT_REF_NBR, MERCHANT_ID, CARD_TYPE, AMOUNT, 
                    PAYMENT_TYPE, APG_ID, APPLICATION_NAME, CARD_SUFFIX, EXPIRY, BIN, TRANSACTION_TIME, 
                    RECONCILIATION_ID, CARD_NBR
                ) SELECT TRANSACTION_DATE, REQUEST_ID, MERCHANT_REF_NBR, MERCHANT_ID, CARD_TYPE, AMOUNT, 
                    PAYMENT_TYPE, APG_ID, APPLICATION_NAME, CARD_SUFFIX, EXPIRY, BIN, TRANSACTION_TIME, 
                    RECONCILIATION_ID, CARD_NBR
                FROM TRUST.CYBERSOURCE AS C WITH (NOLOCK)
                WHERE TRANSACTION_DATE >= ? 
                AND UPPER(PROCESSOR) = 'VDCVANTIV' 
                AND APPLICATION_NAME = 'ICS BILL'
                AND PAYMENT_TYPE = 'AMERICAN EXPRESS'
                AND NOT EXISTS (
                    SELECT RECONCILIATION_ID FROM TRUST.EMAF E WITH (NOLOCK) 
                    WHERE C.RECONCILIATION_ID = E.RECONCILIATION_ID
                )
                AND NOT EXISTS (
                    SELECT AMOUNT FROM TRUST.DISCREPANCY_RESOLUTION DR WITH (NOLOCK) 
                    WHERE SOURCE = 'CS-EMAF'
                    AND CONVERT(CHAR(10), DR.TRANSACTION_DATE, 126) = C.TRANSACTION_DATE
                    AND C.RECONCILIATION_ID = DR.RECONCILIATION_ID
                    AND C.AMOUNT = DR.AMOUNT
                )
                """,
                'parameters': [matchDate]
            }
        }
