# recommender.py

import pandas as pd
import numpy as np

def get_enhanced_product_rankings(postal_code, top_k, model, encoders, processed_data, feature_columns):
    """Get product rankings for a specific outlet using trained LightGBM model"""
    
    if 'Postal Code' not in encoders:
        raise ValueError("Postal Code encoder not found.")

    if postal_code in encoders['Postal Code'].classes_:
        outlet_encoded = encoders['Postal Code'].transform([postal_code])[0]
    else:
        # Default to a known outlet (e.g. most common one)
        default_outlet = encoders['Postal Code'].classes_[0]
        outlet_encoded = encoders['Postal Code'].transform([default_outlet])[0]
        print(f"⚠ Warning: {postal_code} not found. Defaulting to {default_outlet}.")


    unique_products = processed_data['product_id_encoded'].unique()
    pred_data = []

    for product in unique_products:
        record = {
            'postal_code_encoded': outlet_encoded,
            'product_id_encoded': product
        }

        outlet_rows = processed_data[processed_data['postal_code_encoded'] == outlet_encoded]
        if outlet_rows.empty:
            continue

        outlet_data = outlet_rows.iloc[0]
        for col in ['region_encoded', 'city_encoded', 'state_encoded']:
            if col in processed_data.columns:
                record[col] = outlet_data[col]

        product_rows = processed_data[processed_data['product_id_encoded'] == product]
        if product_rows.empty:
            continue

        product_data = product_rows.iloc[0]
        for col in ['category_encoded', 'sub_category_encoded']:
            if col in processed_data.columns:
                record[col] = product_data[col]

        pred_data.append(record)

    pred_df = pd.DataFrame(pred_data)

    # Fill missing features with training set means
    for col in feature_columns:
        if col not in pred_df.columns:
            if col in processed_data.columns:
                pred_df[col] = processed_data[col].mean()
            else:
                pred_df[col] = 0

    pred_df = pred_df[feature_columns].fillna(0)

    # Predict
    predictions = model.predict(pred_df, num_iteration=model.best_iteration)

    ranking_df = pd.DataFrame({
        'product_encoded': unique_products,
        'predicted_potential': predictions
    })

    ranking_df['Product_ID'] = encoders['Product ID'].inverse_transform(unique_products)

    # ➕ Add Product Name (optional but useful)
    if 'Product Name' in encoders:
        product_names = []
        for prod_enc in unique_products:
            prod_data = processed_data[processed_data['product_id_encoded'] == prod_enc]
            if not prod_data.empty:
                name_enc = prod_data['product_name_encoded'].iloc[0]
                name = encoders['Product Name'].inverse_transform([name_enc])[0]
                product_names.append(name)
            else:
                product_names.append('Unknown')
        ranking_df['Product_Name'] = product_names

    if 'Category' in encoders:
        product_categories = []
        for prod_enc in unique_products:
            prod_data = processed_data[processed_data['product_id_encoded'] == prod_enc]
            if not prod_data.empty:
                cat_enc = prod_data['category_encoded'].iloc[0]
                category = encoders['Category'].inverse_transform([cat_enc])[0]
                product_categories.append(category)
            else:
                product_categories.append('Unknown')
        ranking_df['Category'] = product_categories

    historical_sales = []
    for prod in unique_products:
        sales = processed_data[processed_data['product_id_encoded'] == prod]['Sales'].mean()
        historical_sales.append(sales)
    ranking_df['historical_sales'] = historical_sales

    ranking_df = ranking_df.sort_values(
        ['predicted_potential', 'historical_sales'], ascending=[False, False]
    )

    return ranking_df.head(top_k)
