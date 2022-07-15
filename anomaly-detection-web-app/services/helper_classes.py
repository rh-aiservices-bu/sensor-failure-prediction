# Importing libraries
import psycopg2
import pandas as pd

class DBConnection:
    """singleton that connects to a PostgreSQL database"""
    _connection = None
    
    @classmethod 
    def get_connection(cls):
        """reads the database credentials and connects to the database"""
        if cls._connection is None:
            cls._connection = psycopg2.connect(host='postgresql.test-db01.svc.cluster.local',
                                                database='sampledb',
                                                user='user82M',
                                                password='q2s2gLeojWQFMkBa')
            
        return cls._connection

class AnomalyDataService:
    def get_all_data(self, get_all_dataSQL):
        # get_all_dataSQL represents an SQL query statement
        """executes SQL queries to the database"""
        db_conn = DBConnection.get_connection()
        # create a cursor to execute SQL queries
        cursor = db_conn.cursor 
        try:
            result=cursor.execute(get_all_dataSQL)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        else:
            return result
        finally:
            # close connection to the database
            cursor.close()
            
    def get_dfSQL(self):
        """"performs an SQL query on a database table and reads it into a pandas dataframe"""
        db_conn = DBConnection.get_connection()
        df = pd.read_sql_query(get_all_dataSQL, db_conn)
        return df

# Usage example 
query = AnomalyDataService()
get_all_dataSQL = """SELECT * FROM casing1"""
query.get_dfSQL()