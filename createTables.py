from modules import databaseInterface as DBMGR
import configparser

        
# Read from Config file and import api keys
def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

config = read_config()


# Create an instance of the DatabaseManager class
db_manager = DBMGR.DatabaseManager(host=config.get("sql", "host"), database=config.get("sql", "database"), user=config.get("sql", "user"), password=config.get("sql", "password"))


# Create a database connection
db_manager.create_connection()

# Load external SQL script
script_path = 'databaseInit.sql'
with open(script_path, 'r') as file:
    script_content = file.read()

# Split the script content into individual queries
queries = script_content.split(';')

# Execute queries
for query in queries:
    db_manager.execute_query(query)