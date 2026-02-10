import requests
import json
import os

def collect_economic_data():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(project_root, 'data', 'raw', 'economic_data.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # World Bank API indicators: GDP (NY.GDP.MKTP.CD), inflation (FP.CPI.TOTL.ZG), unemployment (SL.UEM.TOTL.ZS)
    indicators = {
        "GDP": "NY.GDP.MKTP.CD",
        "Inflation": "FP.CPI.TOTL.ZG",
        "Unemployment": "SL.UEM.TOTL.ZS"
    }

    countries = ['US', 'GB', 'CN', 'IN']  # Add more country ISO codes as needed
    years = "2015:2024"

    all_data = []

    for indicator_name, indicator_code in indicators.items():
        for country in countries:
            url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator_code}?format=json&date={years}&per_page=100"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    for entry in data[1]:
                        all_data.append({
                            'indicator': indicator_name,
                            'country': country,
                            'date': entry.get('date'),
                            'value': entry.get('value')
                        })
            else:
                print(f"Failed to fetch {indicator_name} data for {country}")

    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=4)

    print(f"Collected economic indicator data for {len(countries)} countries.")

