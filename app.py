from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import nltk
from nltk.metrics import edit_distance
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from flask import Flask, send_from_directory
from flask_caching import Cache
from flask_compress import Compress

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
Compress(app)

file_path = 'Concatenated_dataset.xlsx'
df = pd.read_excel(file_path)

@app.route('/')
@app.route('/landing')
def index():
    try:
        unique_module = df['Module'].unique().tolist()
        sorted_unique_modules=sorted(unique_module)
        # print("Unique modules:", unique_module) 
    except Exception as e:
        print("Error:", e)  
        return "An error occurred."
    
    return render_template('landing.html', unique_module=sorted_unique_modules)

def get_unique_values(column):
    unique_values = column.dropna().unique().tolist()
    return unique_values if unique_values else None

@cache.cached(timeout=60) 
@app.route('/filters', methods=['POST'])
def filters():
    try:
        
        selected_module = request.form.get('module')
        selected_part_keyword = request.form.get('part_keyword')
        selected_shop = request.form.get('shop')
        selected_wspg = request.form.get('wspg')
        selected_sv = request.form.get('SV')
        selected_tech = request.form.get('tech')
        selected_thrust = request.form.get('thrust')
        selected_region = request.form.get('region')
        selected_source = request.form.get('source')
        selected_esn = request.form.get('esn')

        print("Selected Module:", selected_module)
        print("Selected Part Keyword:", selected_part_keyword)
        print("Selected Shop:", selected_shop)
        print("Selected WSPG:", selected_wspg)
        print("Selected SV:", selected_sv)
        print("Selected Tech:", selected_tech)
        print("Selected Thrust:", selected_thrust)
        print("Selected Region:", selected_region)
        print("Selected Source:", selected_source)
        print("Selected ESN:", selected_esn)


        
        if selected_module:
            filtered_df = df[df['Module'] == selected_module]
        else:
            filtered_df = df

    
        if selected_part_keyword:
            filtered_df = filtered_df[filtered_df['SEM Part Keyword_mask'] == selected_part_keyword]

       
        if selected_shop:
            filtered_df = filtered_df[filtered_df['SHOP'] == selected_shop]
        if selected_wspg:
            filtered_df = filtered_df[filtered_df['WSPG'] == selected_wspg]
        if selected_sv:
            filtered_df = filtered_df[filtered_df['SV#'] == selected_sv]
        if selected_tech:
            filtered_df = filtered_df[filtered_df['Tech Insert'] == selected_tech]
        if selected_thrust:
            filtered_df = filtered_df[filtered_df['Thrust'] == selected_thrust]
        if selected_region:
            filtered_df = filtered_df[filtered_df['Region'] == selected_region]
        if selected_source:
            filtered_df = filtered_df[filtered_df['Source'] == selected_source]
        if selected_esn:
            filtered_df = filtered_df[filtered_df['ESN_MASK'] == selected_esn]

        
        response_filters = {
            'part':get_unique_values(filtered_df["SEM Part Keyword_mask"]),
            'shop': get_unique_values(filtered_df["SHOP"]),
            'wspg': get_unique_values(filtered_df["WSPG"]),
            'sv': get_unique_values(filtered_df["SV#"]),
            'tech': get_unique_values(filtered_df["Tech Insert"]),
            'thrust': get_unique_values(filtered_df["Thrust"]),
            'region': get_unique_values(filtered_df["Region"]),
            'source': get_unique_values(filtered_df["Source"]),
            'esn': get_unique_values(filtered_df["ESN_MASK"])
        }

        
        response_filters = {k: v for k, v in response_filters.items() if v is not None}
        filtered_df_filtered = filtered_df[filtered_df['Source'] != 'PROD']
        chart_data1 = {
            'Source': filtered_df_filtered['Source'].dropna().tolist(),
            'Scrap': filtered_df_filtered['Scrap Rate'].dropna().tolist(),
            'Contract': filtered_df_filtered['DERIVED CONTRACT TYPE'].dropna().tolist()
        }

        filtered_df_filtered1=filtered_df_filtered[filtered_df_filtered['Source']=="CC"]
        chart_data2={
            'Operator':filtered_df_filtered1["Operator Name_mask"].dropna().tolist(),
            'SO':filtered_df_filtered1["Parent SO_mask"].dropna().tolist()
        }

        filtered_df_filtered2=filtered_df_filtered[filtered_df_filtered['Source']=="BVD"]
        chart_data22={
            'Operator':filtered_df_filtered2["Operator Name_mask"].dropna().tolist(),
            'SO':filtered_df_filtered2["Parent SO_mask"].dropna().tolist()
        }

        result = filtered_df.groupby('Source').agg(
            Count_WO=('Parent SO_mask', 'count'),
            Avg_Scrap_Rate=('Scrap Rate', 'mean'),
            Avg_ICR_Rate=('ICR Rate', 'mean'),
            Avg_ECR_Rate=('ECR Rate', 'mean')
        ).reset_index()

        nan_counts = result.isna().sum()
        print("NaN values in result DataFrame:", nan_counts)

        for col in ['Avg_Scrap_Rate', 'Avg_ICR_Rate', 'Avg_ECR_Rate']:
            result[col] = result[col].apply(lambda x: round(x, 2) if pd.notna(x) else 0)

        chart_data3 = {
            'Source': result['Source'].tolist(),
            'Count_WO': result['Count_WO'].tolist(),
            'Avg_Scrap_Rate': result['Avg_Scrap_Rate'].tolist(),
            'Avg_ICR_Rate': result['Avg_ICR_Rate'].tolist(),
            'Avg_ECR_Rate': result['Avg_ECR_Rate'].tolist()
        }

        nan_counts_chart_data3 = {key: sum(1 for v in value if v == 0) for key, value in chart_data3.items()}
        print("NaN or 0 values in chart_data3:", nan_counts_chart_data3)



        mean_scrap_by_year_and_source = filtered_df.groupby(['Event_year', 'Source'])['Scrap Rate'].mean().reset_index()
        mean_scrap_by_year_and_source['Scrap Rate'] = mean_scrap_by_year_and_source['Scrap Rate'].apply(lambda x: round(x, 2) if pd.notna(x) else x)
        mean_scrap_by_year_and_source = mean_scrap_by_year_and_source.dropna(subset=['Scrap Rate'])

        prod_avg_scrap_rate = filtered_df[filtered_df['Source'] == 'PROD']['Scrap Rate'].mean()
        prod_avg_scrap_rate = round(prod_avg_scrap_rate, 2) if pd.notna(prod_avg_scrap_rate) else None

        if pd.isna(prod_avg_scrap_rate):
            print("NaN found in prod_avg_scrap_rate")
            unique_event_years = mean_scrap_by_year_and_source['Event_year'].unique() if not mean_scrap_by_year_and_source.empty else []
        else:
            unique_event_years = mean_scrap_by_year_and_source['Event_year'].unique()

        prod_entries = pd.DataFrame({
            'Event_year': unique_event_years,
            'Source': ['PROD'] * len(unique_event_years),
            'Scrap Rate': [prod_avg_scrap_rate] * len(unique_event_years)
        })

        combined_mean_scrap = pd.concat([mean_scrap_by_year_and_source, prod_entries], ignore_index=True)
        combined_mean_scrap = combined_mean_scrap.dropna(subset=['Scrap Rate'])

        scrap_data_unique_years_sources = {
            'Event_year': combined_mean_scrap["Event_year"].tolist(),
            'Source': combined_mean_scrap["Source"].tolist(),
            'Scrap': combined_mean_scrap["Scrap Rate"].tolist()
        }


        mean_icr_by_year_and_source = filtered_df.groupby(['Event_year', 'Source'])['ICR Rate'].mean().reset_index()
        mean_icr_by_year_and_source['ICR Rate'] = mean_icr_by_year_and_source['ICR Rate'].apply(lambda x: round(x, 2) if pd.notna(x) else x)
        mean_icr_by_year_and_source = mean_icr_by_year_and_source.dropna(subset=['ICR Rate'])
        prod_avg_icr_rate = filtered_df[filtered_df['Source'] == 'PROD']['ICR Rate'].mean()
        prod_avg_icr_rate = round(prod_avg_icr_rate, 2) if pd.notna(prod_avg_icr_rate) else None

        if pd.isna(prod_avg_icr_rate):
            print("NaN found in prod_avg_icr_rate")
            unique_event_years = mean_icr_by_year_and_source['Event_year'].unique() if not mean_icr_by_year_and_source.empty else []
        else:
            unique_event_years = mean_icr_by_year_and_source['Event_year'].unique()
        prod_entries = pd.DataFrame({
            'Event_year': unique_event_years,
            'Source': ['PROD'] * len(unique_event_years),
            'ICR': [prod_avg_icr_rate] * len(unique_event_years)
        })
        combined_mean_icr = pd.concat([mean_icr_by_year_and_source.rename(columns={'ICR Rate': 'ICR'}), prod_entries], ignore_index=True)
        combined_mean_icr = combined_mean_icr.dropna(subset=['ICR'])
        icr_data_unique_years_sources = {
            'Event_year': combined_mean_icr["Event_year"].tolist(),
            'Source': combined_mean_icr["Source"].tolist(),
            'ICR': combined_mean_icr["ICR"].tolist()
        }


        
        mean_ecr_by_year_and_source = filtered_df.groupby(['Event_year', 'Source'])['ECR Rate'].mean().reset_index()
        mean_ecr_by_year_and_source['ECR Rate'] = mean_ecr_by_year_and_source['ECR Rate'].apply(lambda x: round(x, 2) if pd.notna(x) else x)
        mean_ecr_by_year_and_source = mean_ecr_by_year_and_source.dropna(subset=['ECR Rate'])
        prod_avg_ecr_rate = filtered_df[filtered_df['Source'] == 'PROD']['ECR Rate'].mean()
        prod_avg_ecr_rate = round(prod_avg_ecr_rate, 2) if pd.notna(prod_avg_ecr_rate) else None

        if pd.isna(prod_avg_ecr_rate):
            print("NaN found in prod_avg_ecr_rate")
            unique_event_years = mean_ecr_by_year_and_source['Event_year'].unique() if not mean_ecr_by_year_and_source.empty else []
        else:
            unique_event_years = mean_ecr_by_year_and_source['Event_year'].unique()
        prod_entries = pd.DataFrame({
            'Event_year': unique_event_years,
            'Source': ['PROD'] * len(unique_event_years),
            'ECR': [prod_avg_ecr_rate] * len(unique_event_years)
        })
        combined_mean_ecr = pd.concat([mean_ecr_by_year_and_source.rename(columns={'ECR Rate': 'ECR'}), prod_entries], ignore_index=True)

        combined_mean_ecr = combined_mean_ecr.dropna(subset=['ECR'])
        ecr_data_unique_years_sources = {
            'Event_year': combined_mean_ecr["Event_year"].tolist(),
            'Source': combined_mean_ecr["Source"].tolist(),
            'ECR': combined_mean_ecr["ECR"].tolist()
        }


    


        response = {
            "filters": response_filters,
            "chart": chart_data1,
            "funnel1":chart_data2,
            "funnel2":chart_data22,
            "table":chart_data3,
            'scrap':scrap_data_unique_years_sources,
            'icr':icr_data_unique_years_sources,
            'ecr':ecr_data_unique_years_sources
        }


        def check_for_null_values(response):
            null_entries = {}
            
            for key, value in response.items():
                if value is None or (isinstance(value, (list, dict)) and not value):
                    null_entries[key] = value
                    
            return null_entries
        null_values = check_for_null_values(response)

        if null_values:
            print("Null or empty entries found:", null_values)
        else:
            print("No null or empty entries in the response.")
                

        return jsonify(response)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "An error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run()
