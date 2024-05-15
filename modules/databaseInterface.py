import mysql.connector
from mysql.connector import Error

class DatabaseManager:
    def __init__(self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.conn = None

    def create_connection(self):
        try:
            self.conn = mysql.connector.connect(host=self.host,
                                                database=self.database,
                                                user=self.user,
                                                password=self.password)
            if self.conn.is_connected():
                print('Connected to MySQL database')

        except Error as e:
            print(f"Error: '{e}'")

    def close_connection(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print('Connection closed.')

    def insert_into_table(self, insert_table_sql, data_tuple):
        try:
            cursor = self.conn.cursor()
            cursor.execute(insert_table_sql, data_tuple)
            self.conn.commit()
        except Error as e:
            print(f"Error: '{e}'")

    def select_from_table(self, select_table_sql, data_tuple=None):
        try:
            cursor = self.conn.cursor()
            if data_tuple:
                cursor.execute(select_table_sql, data_tuple)
            else:
                cursor.execute(select_table_sql)
            return cursor.fetchall()
        except Error as e:
            print(f"Error: '{e}'")

    def prepare_insert_sql(self, table, obj):
        keys = obj.keys()
        values = tuple(obj.values())
        sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join(['%s'] * len(values))})"
        return sql, values

    def execute_query(self, query):
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
            print('Query executed successfully')

        except Error as e:
            print(f"Error: '{e}'")