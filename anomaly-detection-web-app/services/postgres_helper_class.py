import psycopg2
import pandas as pd
from configparser import ConfigParser

class Postgres:
    """A helper class to connect to a PostgreSQL database and perform SQL queries on it"""
    __instance = None
    
    def __new__(cls, filename='database.ini', section='postgresql'):
        """checks for pre-exsisting connection to database"""
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
              
            parser = ConfigParser()
            parser.read(filename)
            
            db={}
            if parser.has_section(section):
                params = parser.items(section)
                for param in params:
                    db[param[0]] = param[1]
            else:
                raise Exception('Section {0} not found in the {1} file'.format(section, filename))
                
            try:
                connect = Postgres.__instance.connect = psycopg2.connect(**db)
                cursor = Postgres.__instance.cursor = connect.cursor()
                
                # execute a statement
                print('PostgreSQL database version:')
                cursor.execute('SELECT version()')

                # display the PostgreSQL database server version
                db_version = cursor.fetchone()
                print(db_version)
                
            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
        return cls.__instance
                    
    def __init__(self):
        """reads the configuration file and returns connection parameters"""
        self.connect = self.__instance.connect
        self.cursor = self.__instance.cursor
        
    def query(self, query, params = None):
        """executes SQL queries to the database"""
        try:
            result = self.cursor.execute(query, params)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        else:
            return result
    
    def commit(self):
        """commits changes to the database"""
        self.connect.commit()
        
    def create_pandas_df(self, query):
        """performs a SQL query on a database table and reads it into a pandas dataframe"""
        df = pd.read_sql_query(query, self.connect)
        return df
            
    def close(self): 
        """close connection to the database"""
        self.cursor.close()
        self.connect.close() 