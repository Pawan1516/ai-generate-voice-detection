"""
Test script to fetch and inspect Hugging Face dataset structure
"""
import requests
import json

def test_huggingface_api():
    """Test the Hugging Face dataset API"""
    url = "https://datasets-server.huggingface.co/rows?dataset=Oration%2Fvoice-activity-detection&config=default&split=train&offset=0&length=2"
    
    print("Fetching data from Hugging Face...")
    print(f"URL: {url}\n")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print("API Response Structure:")
        print(f"Keys: {list(data.keys())}\n")
        
        if 'features' in data:
            print("Features:")
            for feature in data['features']:
                print(f"  - {feature.get('name')}: {feature.get('_type')}")
            print()
        
        if 'rows' in data:
            print(f"Number of rows: {len(data['rows'])}\n")
            
            if len(data['rows']) > 0:
                print("First row structure:")
                first_row = data['rows'][0]
                print(f"Keys: {list(first_row.keys())}\n")
                
                if 'row' in first_row:
                    row_data = first_row['row']
                    print("Row data keys:")
                    for key in row_data.keys():
                        value = row_data[key]
                        if isinstance(value, dict):
                            print(f"  - {key}: dict with keys {list(value.keys())}")
                        elif isinstance(value, list):
                            print(f"  - {key}: list with {len(value)} items")
                        else:
                            print(f"  - {key}: {type(value).__name__}")
                    
                    print("\nSample row data:")
                    print(json.dumps(first_row, indent=2, default=str)[:500] + "...")
        
        print("\n✅ API connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_huggingface_api()
