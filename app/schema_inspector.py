from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.engine import reflection
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import json

# Load environment variables
load_dotenv()

def get_database_url() -> str:
    """Get database URL from environment variables."""
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '3306'),
        'database': os.getenv('DB_NAME', 'recruitment_crm'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
    }
    return f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

def list_tables() -> List[str]:
    """List all tables in the database."""
    try:
        engine = create_engine(get_database_url())
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return tables
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return []

def get_table_structure(table_name: str) -> Dict[str, Any]:
    """Get the structure of a specific table."""
    try:
        engine = create_engine(get_database_url())
        inspector = inspect(engine)
        
        # Get columns
        columns = inspector.get_columns(table_name)
        column_info = []
        for column in columns:
            column_info.append({
                'name': column['name'],
                'type': str(column['type']),
                'nullable': column['nullable'],
                'primary_key': column.get('primary_key', False)
            })
        
        # Get foreign keys
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        return {
            'columns': column_info,
            'foreign_keys': foreign_keys
        }
    except Exception as e:
        print(f"Error getting table structure: {str(e)}")
        return {}

if __name__ == "__main__":
    # First, list all tables
    print("\n=== Available Tables ===")
    tables = list_tables()
    if tables:
        print("\nFound the following tables:")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")
        
        # Ask user which table to inspect
        print("\nEnter the number of the table you want to inspect (or 'all' for all tables):")
        choice = input("> ").strip().lower()
        
        if choice == 'all':
            for table in tables:
                print(f"\n=== Structure of {table} ===")
                structure = get_table_structure(table)
                print(json.dumps(structure, indent=2))
        elif choice.isdigit() and 1 <= int(choice) <= len(tables):
            table = tables[int(choice) - 1]
            print(f"\n=== Structure of {table} ===")
            structure = get_table_structure(table)
            print(json.dumps(structure, indent=2))
        else:
            print("Invalid choice!")
    else:
        print("No tables found in the database.") 