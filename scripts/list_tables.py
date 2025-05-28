from app.schema_inspector import list_tables, get_table_structure
import json

if __name__ == "__main__":
    # List all tables
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