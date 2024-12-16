from logging import Logger
from DBLoader import DBLoader


class CardPayment(DBLoader):
    def __init__(self, name, log: Logger, startDate, endDate) -> None:
        super().__init__(name, log, startDate, endDate)
        self.matching_tables_to_clean = ['CS_CARDPAYMENT', 'CARDPAYMENT_DMS']
        self.stat_queries = {
            self.UNMATCHED_STATS: """
                SELECT 
                    'CARDPAYMENT' AS SOURCE, TRANSACTION_DATE, MERCHANT_ID, 
                    SUM(AMOUNT) AS AMOUNT, COUNT(*) AS COUNT 
                FROM TRUST.CARDPAYMENT WITH (NOLOCK) 
                GROUP BY TRANSACTION_DATE, MERCHANT_ID
            """,
            self.UNMATCHED: """
                SELECT 
                    'CS-CARDPAYMENT' AS SOURCE, TRANSACTION_DATE, MERCHANT_ID, 
                    AMOUNT, TRANSACTION_TIME, REQUEST_ID, MERCHANT_REF_NBR, 
                    '' AS RECONCILIATION_ID, '' AS DMS_FINANCIAL_ID, 
                    0 AS APG_ID, '' AS APP_NAME 
                FROM TRUST.CS_CARDPAYMENT WITH (NOLOCK)
                UNION ALL
                SELECT 
                    'CARDPAYMENT-DMS' AS SOURCE, TRANSACTION_DATE, MERCHANT_ID, 
                    AMOUNT, TRANSACTION_TIME, REQUEST_ID, MERCHANT_REF_NBR, 
                    '' AS RECONCILIATION_ID, '' AS DMS_FINANCIAL_ID, 
                    0 AS APG_ID, '' AS APP_NAME 
                FROM TRUST.CARDPAYMENT_DMS WITH (NOLOCK)
            """,
            self.STATS: """
                SELECT 
                    TRANSACTION_DATE, 'CARDPAYMENT' AS SOURCE, 
                    COUNT(*) AS CNT, SUM(AMOUNT) AS AMOUNT 
                FROM TRUST.CARDPAYMENT WITH (NOLOCK) 
                GROUP BY TRANSACTION_DATE
            """
        }

    def load(self):
        recordCount = 0
        totalAmount = 0

        self.log.info(
            f"Started CARDPAYMENT load from DATADB for {self.startDate} to but not including {self.endDate}"
        )

        # Establish database connections
        connDataDb = self.db_conn(
            self.sql_datastore_server, self.sql_datastore_database,
            self.sql_datastore_username, self.sql_datastore_password
        )
        conn = self.db_conn(
            self.sql_server, self.sql_working_database,
            self.sql_working_username, self.sql_working_password
        )

        cursorDataDb = connDataDb.cursor()
        cursor = conn.cursor()

        try:
            cursor.fast_executemany = True

            # Query to insert data
            sql = """
                INSERT INTO TRUST.CARDPAYMENT (
                    AMOUNT, CARD_TYPE, PAYMENT_TYPE, MERCHANT_ID, 
                    MERCHANT_REF_NBR, REQUEST_ID, TRANSACTION_DATE, 
                    CARD_SUFFIX, BIN, TRANSACTION_TIME, TRANSACTION_ID
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            # Query to select data
            selectSql = """
                SELECT
                    AMOUNT AS AMOUNT,
                    CASE
                        WHEN CARDBRAND = 'VISA' THEN 'VISA'
                        WHEN CARDBRAND = 'MASTERCARD' THEN 'MCRD'
                        WHEN CARDBRAND = 'AMERICANEXPRESS' THEN 'AMEX'
                        WHEN CARDBRAND = 'DISCOVER' THEN 'DISC'
                        ELSE 'OTHER'
                    END AS CARD_TYPE,
                    CARDBRAND AS PAYMENT_TYPE,
                    ALSACMERCHANTID AS MERCHANT_ID,
                    TRANSACTIONKEY AS MERCHANT_REF_NBR,
                    PROCESSORTRANSACTIONID AS REQUEST_ID,
                    CONVERT(CHAR(10), DATECREATED, 126) AS TRANSACTION_DATE,
                    CARDLASTFOUR AS CARD_SUFFIX,
                    CARDBIN AS BIN,
                    RIGHT(CONVERT(CHAR(19), DATECREATED, 120), 8) AS TRANSACTION_TIME,
                    clientTransactionId AS TRANSACTION_ID
                FROM CARDPAYMENT.TRANSACTIONS WITH (NOLOCK)
                WHERE DATECREATED >= ?
                AND DATECREATED < ?
                AND PROCESSORRESPONSETEXT = 'AUTHORIZED'
                ORDER BY DATECREATED ASC
            """

            cursorDataDb.execute(selectSql, [self.startDate, self.endDate])

            # Process rows
            tuples = []
            for row in cursorDataDb:
                tuple = [
                    float(row.AMOUNT),
                    row.CARD_TYPE,
                    row.PAYMENT_TYPE,
                    row.MERCHANT_ID,
                    row.MERCHANT_REF_NBR,
                    row.REQUEST_ID,
                    row.TRANSACTION_DATE,
                    row.CARD_SUFFIX,
                    row.BIN,
                    row.TRANSACTION_TIME,
                    row.TRANSACTION_ID
                ]
                recordCount += 1
                totalAmount += row.AMOUNT
                tuples.append(tuple)

                if len(tuples) >= self.sql_batch_size:
                    cursor.executemany(sql, tuples)
                    conn.commit()
                    tuples = []

            # Insert any remaining rows
            if tuples:
                cursor.executemany(sql, tuples)
                conn.commit()

            self.log.info(
                f"Finished CARDPAYMENT load. Records: {recordCount}, Amount: {totalAmount:,.2f}"
            )

        except Exception as e:
            conn.rollback()
            self.log.error(f"Error inserting records into database: {repr(e)}")

        finally:
            cursorDataDb.close()
            cursor.close()
            connDataDb.close()
            conn.close()

    def get_matchers(self, matchDate):
        return {
            'CyberSource->CARDPAYMENT': {
                'sql': """
                    INSERT INTO TRUST.CS_CARDPAYMENT (
                        TRANSACTION_DATE, REQUEST_ID, MERCHANT_REF_NBR, MERCHANT_ID, 
                        CARD_TYPE, AMOUNT, PAYMENT_TYPE, APG_ID, APPLICATION_NAME, 
                        CARD_SUFFIX, EXPIRY, BIN, TRANSACTION_TIME, RECONCILIATION_ID, CARD_NBR
                    )
                    SELECT TRANSACTION_DATE, REQUEST_ID, MERCHANT_REF_NBR, MERCHANT_ID, 
                        CARD_TYPE, AMOUNT, PAYMENT_TYPE, APG_ID, APPLICATION_NAME, 
                        CARD_SUFFIX, EXPIRY, BIN, TRANSACTION_TIME, RECONCILIATION_ID, CARD_NBR
                    FROM TRUST.CYBERSOURCE WITH (NOLOCK)
                    WHERE TRANSACTION_DATE = ?
                    AND NOT EXISTS (
                        SELECT REQUEST_ID FROM TRUST.CARDPAYMENT WITH (NOLOCK)
                        WHERE REQUEST_ID = TRUST.CYBERSOURCE.REQUEST_ID
                    )
                """,
                'parameters': [matchDate]
            },
            'CARDPAYMENT->DMS': {
                'sql': """
                    WITH dms AS (
                        SELECT DMS_FINANCIAL_ID
                        FROM TRUST.DMS WITH (NOLOCK)
                        WHERE (TRANSACTION_DATE = ? OR POSTDATE = ?)
                        AND PAYMENTMETHODCODE = 2
                    )
                    INSERT INTO TRUST.CARDPAYMENT_DMS (
                        TRANSACTION_DATE, REQUEST_ID, TRANSACTION_ID, MERCHANT_REF_NBR, 
                        MERCHANT_ID, CARD_TYPE, AMOUNT, PAYMENT_TYPE, CARD_SUFFIX, 
                        BIN, TRANSACTION_TIME
                    )
                    SELECT 
                        TRANSACTION_DATE, REQUEST_ID, TRANSACTION_ID, MERCHANT_REF_NBR, 
                        MERCHANT_ID, CARD_TYPE, AMOUNT, PAYMENT_TYPE, CARD_SUFFIX, 
                        BIN, TRANSACTION_TIME
                    FROM TRUST.CARDPAYMENT WITH (NOLOCK)
                    LEFT JOIN dms ON TRUST.CARDPAYMENT.TRANSACTION_ID = dms.DMS_FINANCIAL_ID
                    WHERE TRANSACTION_DATE = ? AND dms.DMS_FINANCIAL_ID IS NULL
                """,
                'parameters': [matchDate, matchDate, matchDate]
            }
        }
