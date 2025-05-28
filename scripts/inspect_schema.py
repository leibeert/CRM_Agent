from app.schema_inspector import inspect_schema, print_schema_info, save_schema_to_file

if __name__ == "__main__":
    try:
        # Inspect the schema
        schema_info = inspect_schema()
        
        # Print the schema information
        print_schema_info(schema_info)
        
        # Save to file
        save_schema_to_file(schema_info)
        print(f"\nSchema information has been saved to schema.json")
        
    except Exception as e:
        print(f"Error inspecting schema: {str(e)}") 