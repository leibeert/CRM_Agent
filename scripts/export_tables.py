import pandas as pd
import pymysql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'recruitment_crm')
}

def get_all_tables(conn):
    """Get list of all tables in the database."""
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]

def export_table_to_csv(conn, table_name, output_dir):
    """Export a single table to CSV."""
    try:
        # Read table into pandas DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        output_file = os.path.join(output_dir, f"{table_name}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Successfully exported {table_name} to {output_file}")
        
    except Exception as e:
        print(f"Error exporting {table_name}: {str(e)}")

def main():
    # Create output directory
    output_dir = "database_tables"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Connect to database
        conn = pymysql.connect(**DB_CONFIG)
        print("Connected to database successfully!")
        
        # Get all tables
        tables = get_all_tables(conn)
        print(f"\nFound {len(tables)} tables:")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")
        
        # Export each table
        print("\nExporting tables to CSV files...")
        for table in tables:
            export_table_to_csv(conn, table, output_dir)
        
        print(f"\nAll tables have been exported to the '{output_dir}' directory")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        if 'conn' in locals():
            conn.close()
            print("\nDatabase connection closed")

if __name__ == "__main__":
    main() 