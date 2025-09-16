import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

# --- Step 1: Understand the data (Conceptual) ---
# This is a placeholder to document the plan for identifying and gathering relevant datasets.
# In a real-world scenario, this step would involve
# consultations with domain experts and IT teams to identify actual data sources
# and access methods.

# Potential Data Sources and Relevant Information:

# 1. Vendor Data:
#    - Potential Source: Vendor Management System (VMS) database, ERP system.
#    - Relevant Tables/Files: Vendor master data, vendor performance records, contract information.
#    - Extraction Method: Database queries (SQL), API calls, data exports (CSV, XML).

# 2. Supply Chain Data:
#    - Potential Source: ERP system, Logistics Management System (LMS), Warehouse Management System (WMS).
#    - Relevant Tables/Files: Purchase orders, shipping records, goods receipts, inventory movements.
#    - Extraction Method: Database queries (SQL), API calls, data exports.

# 3. Warranty Claims Data:
#    - Potential Source: Maintenance Management System (MMS), dedicated Warranty Tracking System.
#    - Relevant Tables/Files: Warranty claims details, repair history, part replacements.
#    - Extraction Method: Database queries (SQL), data exports.

# 4. Inspection Reports Data:
#    - Potential Source: Quality Management System (QMS), Inspection Reporting System, potentially document management systems.
#    - Relevant Tables/Files: Inspection results, defect reports, compliance checks.
#    - Extraction Method: Database queries (SQL), file system access for reports (PDF, scanned documents - might require OCR), API calls.

# 5. Support Inventory Management Data:
#    - Potential Source: WMS, ERP system, Asset Management System.
#    - Relevant Tables/Files: Current inventory levels, stock movements, reorder points, spare parts lists.
#    - Extraction Method: Database queries (SQL), API calls, data exports.

print("Step 1: Data source identification and planning steps documented (Conceptual).")

# --- Step 2: Data preprocessing and feature engineering ---

# Creating dummy DataFrames to simulate loaded data
# Vendor Data
vendor_data = {
    'vendor_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'vendor_name': ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D', 'Vendor E', 'Vendor A', 'Vendor B', 'Vendor C', 'Vendor D', 'Vendor E'],
    'contract_start_date': pd.to_datetime(['2020-01-01', '2019-05-15', '2021-03-20', '2018-11-10', '2022-07-01', '2020-01-01', '2019-05-15', '2021-03-20', '2018-11-10', '2022-07-01']),
    'contract_end_date': pd.to_datetime(['2023-12-31', '2024-12-31', '2025-12-31', '2023-12-31', '2026-12-31', '2023-12-31', '2024-12-31', '2025-12-31', '2023-12-31', '2026-12-31']),
    'on_time_delivery_rate': [0.95, 0.92, 0.98, 0.85, 0.99, 0.96, 0.91, 0.97, 0.88, 0.99],
    'quality_score': [4.5, 4.2, 4.8, 3.5, 4.9, 4.6, 4.3, 4.7, 3.8, 4.9],
    'response_time_hours': [24, 36, 12, 48, 8, 22, 38, 10, 45, 7]
}
df_vendors = pd.DataFrame(vendor_data)

# Supply Chain Data
supply_chain_data = {
    'order_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'vendor_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'part_id': ['A1', 'B2', 'C3', 'D4', 'E5', 'A1', 'B2', 'C3', 'D4', 'E5'],
    'order_date': pd.to_datetime(['2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25', '2023-01-30', '2023-02-05', '2023-02-10', '2023-02-15', '2023-02-20', '2023-02-25']),
    'delivery_date': pd.to_datetime(['2023-02-01', '2023-02-10', '2023-02-18', '2023-03-05', '2023-02-28', '2023-03-01', '2023-03-08', '2023-03-12', '2023-03-28', '2023-03-20']),
    'quantity_ordered': [100, 150, 200, 50, 300, 120, 160, 210, 60, 320],
    'quantity_received': [100, 148, 200, 45, 300, 120, 158, 210, 58, 320]
}
df_supply_chain = pd.DataFrame(supply_chain_data)

# Warranty Claims Data
warranty_claims_data = {
    'claim_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'part_id': ['A1', 'B2', 'C3', 'D4', 'E5', 'A1', 'B2', 'C3', 'D4', 'E5'],
    'claim_date': pd.to_datetime(['2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20', '2024-01-25', '2024-02-01', '2024-02-05', '2024-02-10', '2024-02-15', '2024-02-20']),
    'failure_type': ['Mechanical', 'Electrical', 'Wear and Tear', 'Mechanical', 'Other', 'Electrical', 'Mechanical', 'Wear and Tear', 'Electrical', 'Other'],
    'resolution_date': pd.to_datetime(['2024-01-20', '2024-01-25', '2024-01-30', '2024-02-05', '2024-02-10', '2024-02-15', '2024-02-20', '2024-02-25', '2024-03-01', '2024-03-05']),
    'resolution_status': ['Repaired', 'Replaced', 'Repaired', 'Replaced', 'No Issue Found', 'Replaced', 'Repaired', 'Repaired', 'Replaced', 'No Issue Found']
}
df_warranty = pd.DataFrame(warranty_claims_data)

# Inspection Reports Data
inspection_reports_data = {
    'inspection_id': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
    'part_id': ['A1', 'B2', 'C3', 'D4', 'E5', 'A1', 'B2', 'C3', 'D4', 'E5'],
    'inspection_date': pd.to_datetime(['2023-03-01', '2023-03-05', '2023-03-10', '2023-03-15', '2023-03-20', '2023-04-01', '2023-04-05', '2023-04-10', '2023-04-15', '2023-04-20']),
    'inspector_id': [10, 11, 12, 10, 11, 12, 10, 11, 12, 10],
    'inspection_result': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass'],
    'defect_category': [np.nan, 'Dimensional', np.nan, 'Surface', np.nan, np.nan, 'Functional', np.nan, 'Dimensional', np.nan]
}
df_inspections = pd.DataFrame(inspection_reports_data)

# Inventory Management Data
inventory_management_data = {
    'inventory_id': [3001, 3002, 3003, 3004, 3005],
    'part_id': ['A1', 'B2', 'C3', 'D4', 'E5'],
    'stock_level': [500, 300, 700, 150, 1000],
    'reorder_point': [100, 50, 200, 30, 250],
    'last_restock_date': pd.to_datetime(['2023-04-15', '2023-04-20', '2023-04-10', '2023-04-25', '2023-04-05'])
}
df_inventory = pd.DataFrame(inventory_management_data)

print("\nDummy dataframes created.")

# Perform initial data cleaning steps
df_vendors.drop_duplicates(inplace=True)
print("\ndf_vendors after removing duplicates:")
display(df_vendors.head())

df_supply_chain['lead_time'] = (df_supply_chain['delivery_date'] - df_supply_chain['order_date']).dt.days
df_supply_chain['delivery_variance'] = df_supply_chain['quantity_ordered'] - df_supply_chain['quantity_received']
df_supply_chain['quantity_received'].fillna(0, inplace=True)
print("\ndf_supply_chain after cleaning and initial feature creation:")
display(df_supply_chain.head())

df_warranty['resolution_date'].fillna(pd.to_datetime('today'), inplace=True)
df_warranty['resolution_time_days'] = (df_warranty['resolution_date'] - df_warranty['claim_date']).dt.days
print("\ndf_warranty after cleaning and initial feature creation:")
display(df_warranty.head())

df_inspections['defect_category'].fillna('No Defect', inplace=True)
df_inspections['inspection_result_numeric'] = df_inspections['inspection_result'].apply(lambda x: 1 if x == 'Pass' else 0)
print("\ndf_inspections after cleaning and initial feature creation:")
display(df_inspections.head())

print("\nInitial data cleaning complete.")

# Create relevant features for each domain
vendor_performance = df_vendors.groupby('vendor_id').agg(
    avg_on_time_delivery_rate=('on_time_delivery_rate', 'mean'),
    avg_quality_score=('quality_score', 'mean'),
    avg_response_time_hours=('response_time_hours', 'mean')
).reset_index()
print("\nAggregated Vendor Performance Features:")
display(vendor_performance.head())

supply_chain_agg = df_supply_chain.groupby(['part_id', 'vendor_id']).agg(
    avg_lead_time=('lead_time', 'mean'),
    total_delivery_variance=('delivery_variance', 'sum'),
    order_count=('order_id', 'count')
).reset_index()
print("\nAggregated Supply Chain Features:")
display(supply_chain_agg.head())

warranty_agg = df_warranty.groupby('part_id').agg(
    total_claims=('claim_id', 'count'),
    avg_resolution_time_days=('resolution_time_days', 'mean'),
    failure_type_counts=('failure_type', lambda x: x.value_counts().to_dict())
).reset_index()
print("\nAggregated Warranty Features:")
display(warranty_agg.head())

inspection_agg = df_inspections.groupby('part_id').agg(
    total_inspections=('inspection_id', 'count'),
    failure_count=('inspection_result_numeric', lambda x: (x == 0).sum()),
    defect_category_counts=('defect_category', lambda x: x.value_counts().to_dict())
).reset_index()
inspection_agg['failure_rate'] = inspection_agg['failure_count'] / inspection_agg['total_inspections']
print("\nAggregated Inspection Features:")
display(inspection_agg.head())

inventory_features = df_inventory.copy()
inventory_features['inventory_turnover_rate_placeholder'] = np.nan
print("\nInventory Features (Placeholder for Turnover):")
display(inventory_features.head())

print("\nFeature engineering complete.")

# Consider merging or combining data from different domains
merged_supply_vendor = pd.merge(supply_chain_agg, vendor_performance, on='vendor_id', how='left')
merged_supply_vendor_inspection = pd.merge(merged_supply_vendor, inspection_agg, on='part_id', how='left')
final_merged_data = pd.merge(merged_supply_vendor_inspection, warranty_agg, on='part_id', how='left')

print("\nData merging complete. Final Merged Data Head:")
display(final_merged_data.head())

print("\nStep 2: Data preprocessing and feature engineering complete.")

# --- Step 3: Define the prediction tasks ---
print("\n--- Step 3: Defined Prediction Tasks ---")
print("1. Vendor Performance Prediction: Predict a vendor's future performance (Regression/Classification).")
print("2. Supply Chain Disruption Prediction: Predict potential future delivery delays or quantity discrepancies (Regression/Classification/Time Series/Anomaly Detection).")
print("3. Warranty Claim Frequency/Type Prediction: Predict the number of warranty claims or failure type for parts (Regression/Classification/Time Series).")
print("4. Inspection Failure Rate Prediction: Predict the likelihood of a part failing inspection or the most likely defect category (Classification/Regression/Multi-class Classification).")
print("5. Optimal Inventory Level / Stockout Prediction: Predict optimal stock levels or the risk of stockouts for parts (Regression/Classification/Time Series - requires demand data).")
print("-" * 50)

# --- Step 4: Select appropriate AI models ---
print("\n--- Step 4: Selected AI Models ---")
print("1. Vendor Performance Prediction: Regression (Linear, Gradient Boosting), Classification (Logistic Regression, Random Forest).")
print("2. Supply Chain Disruption Prediction: Regression/Classification (Linear/Logistic Regression, Random Forest, Gradient Boosting), Time Series (ARIMA, Prophet), Anomaly Detection (Isolation Forest).")
print("3. Warranty Claim Frequency/Type Prediction: Regression (Poisson, Negative Binomial), Classification (Logistic Regression, Random Forest), Time Series (ARIMA, Prophet).")
print("4. Inspection Failure Rate Prediction: Classification (Logistic Regression, SVM, Random Forest), Regression (Linear, Beta Regression), Multi-class Classification (Logistic Regression, Random Forest).")
print("5. Optimal Inventory Level / Stockout Prediction: Regression (Linear), Classification (Logistic Regression, Random Forest), Time Series (ARIMA, Prophet, Exponential Smoothing).")
print("-" * 50)


# --- Step 5: Train and evaluate models ---

# Helper function to handle data preparation, model training, and evaluation for regression
def train_evaluate_regression(df, features, target, task_name):
    print(f"\n--- {task_name} (Regression) ---")
    X = df[features]
    y = df[target]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create a regression pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

    # Split data
    # Adjust test_size for small datasets
    test_size = 0.4 if len(df) * 0.2 < 2 else 0.2 # Ensure at least 2 samples in test set if possible
    if len(df) < 2: # Handle extremely small datasets where splitting is not possible
         print("  Dataset too small for splitting. Skipping training and evaluation.")
         return None, float('nan'), float('nan')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float('nan')


    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    return model, mse, r2

# Helper function to handle data preparation, model training, and evaluation for classification
def train_evaluate_classification(df, features, target, task_name):
    print(f"\n--- {task_name} (Classification) ---")
    X = df[features]
    y = df[target]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Determine multi_class strategy
    multi_class_strategy = 'auto' if len(y.unique()) <= 2 else 'ovr'

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='liblinear', multi_class=multi_class_strategy))])

    # Split data
    # Adjust test_size for small datasets
    test_size = 0.4 if len(df) * 0.2 < 2 else 0.2 # Ensure at least 2 samples in test set if possible
    # Use stratification only if test set size is >= number of classes and there is more than one class
    stratify_y = y if (len(y.unique()) > 1 and len(df) * test_size >= len(y.unique())) else None

    if len(df) < 2: # Handle extremely small datasets where splitting is not possible
         print("  Dataset too small for splitting. Skipping training and evaluation.")
         return None, "N/A", "N/A", "N/A", "N/A", "N/A"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_y)

    # Train model only if training set has at least one sample and test set has at least one sample with more than one class (for meaningful eval)
    if len(X_train) > 0 and len(X_test) > 0 and (len(y_test.unique()) > 1 or stratify_y is None):
        try:
            # Check if there are at least two classes in the training data for classification
            if len(y_train.unique()) < 2:
                 print("  Cannot train classification model: Training data contains only a single class.")
                 return None, "N/A", "N/A", "N/A", "N/A", "N/A"

            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # Precision, Recall, F1 need average parameter for multi-class
            average_metric = 'binary' if len(y.unique()) <= 2 else 'weighted'
            precision = precision_score(y_test, y_pred, average=average_metric, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average_metric)
            f1 = f1_score(y_test, y_pred, average=average_metric)

            # ROC-AUC calculation depending on number of classes in the test set
            roc_auc = "N/A"
            if len(y_test.unique()) > 1:
                 if hasattr(model, 'predict_proba') and model.predict_proba(X_test).shape[1] > 1:
                     if len(y.unique()) > 2: # Multi-class ROC AUC
                           try:
                               roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
                           except ValueError:
                               roc_auc = "N/A (Could not compute multi-class ROC-AUC)"
                     else: # Binary ROC AUC
                           roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                 else:
                     roc_auc = "N/A (Predict_proba not available or insufficient classes in test set)"


            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision ({average_metric}): {precision:.4f}")
            print(f"  Recall ({average_metric}): {recall:.4f}")
            print(f"  F1-score ({average_metric}): {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc}")
            return model, accuracy, precision, recall, f1, roc_auc

        except Exception as e:
            print(f"  Error during training or evaluation: {e}")
            return None, "Error", "Error", "Error", "Error", "Error"

    else:
        print("  Cannot train or evaluate: Training or test set is empty or test set contains only a single class and stratification was not possible.")
        return None, "N/A", "N/A", "N/A", "N/A", "N/A"


# 1. Vendor Performance Prediction (Predicting avg_quality_score - Regression)
vendor_features = ['avg_lead_time', 'total_delivery_variance', 'order_count', 'avg_on_time_delivery_rate', 'avg_response_time_hours']
vendor_target = 'avg_quality_score'
vendor_regression_model, vendor_mse, vendor_r2 = train_evaluate_regression(final_merged_data, vendor_features, vendor_target, "Vendor Performance Prediction")

# 2. Supply Chain Disruption Prediction (Predicting likelihood of delivery variance > 0 - Classification)
df_supply_chain['is_disrupted'] = (df_supply_chain['delivery_variance'] > 0).astype(int)
supply_features = ['vendor_id', 'part_id', 'quantity_ordered', 'quantity_received', 'lead_time']
supply_target = 'is_disrupted'
supply_classification_model, supply_accuracy, supply_precision, supply_recall, supply_f1, supply_roc_auc = train_evaluate_classification(df_supply_chain, supply_features, supply_target, "Supply Chain Disruption Prediction")

# 3. Warranty Claim Frequency Prediction (Predicting total_claims - Regression)
warranty_freq_features = ['failure_rate', 'avg_resolution_time_days']
warranty_freq_target = 'total_claims'
warranty_freq_regression_model, warranty_freq_mse, warranty_freq_r2 = train_evaluate_regression(final_merged_data, warranty_freq_features, warranty_freq_target, "Warranty Claim Frequency Prediction")

# Warranty Claim Type Prediction (Predicting primary failure type - Classification)
df_warranty['failure_type_encoded'] = df_warranty['failure_type'].astype('category').cat.codes
warranty_type_features_clf = ['part_id', 'resolution_status', 'resolution_time_days']
warranty_type_target_encoded = 'failure_type_encoded'
warranty_type_classification_model, accuracy_wt, precision_wt, recall_wt, f1_wt, roc_auc_wt = train_evaluate_classification(df_warranty, warranty_type_features_clf, warranty_type_target_encoded, "Warranty Claim Type Prediction")

# 4. Inspection Failure Rate Prediction (Predicting failure_rate - Regression)
inspection_rate_features = ['total_inspections', 'total_claims', 'avg_resolution_time_days']
inspection_rate_target = 'failure_rate'
inspection_rate_regression_model, inspection_rate_mse, inspection_rate_r2 = train_evaluate_regression(final_merged_data, inspection_rate_features, inspection_rate_target, "Inspection Failure Rate Prediction")

# Inspection Failure (Predicting inspection_result (Fail=1) - Classification)
df_inspections['inspection_result_numeric'] = df_inspections['inspection_result'].apply(lambda x: 1 if x == 'Fail' else 0)
inspection_features_clf = ['part_id', 'inspector_id', 'defect_category']
inspection_target_clf = 'inspection_result_numeric'
inspection_classification_model, inspection_accuracy, inspection_precision, inspection_recall, inspection_f1, inspection_roc_auc = train_evaluate_classification(df_inspections, inspection_features_clf, inspection_target_clf, "Inspection Failure Prediction")

# 5. Optimal Inventory Level / Stockout Prediction (Predicting Stockout Risk - Classification)
df_inventory['stockout_risk'] = (df_inventory['stock_level'] < df_inventory['reorder_point']).astype(int)
inventory_features_clf = ['part_id', 'stock_level', 'reorder_point']
inventory_target_clf = 'stockout_risk'
inventory_classification_model, inventory_accuracy, inventory_precision, inventory_recall, inventory_f1, inventory_roc_auc = train_evaluate_classification(df_inventory, inventory_features_clf, inventory_target_clf, "Stockout Risk Prediction")

print("\nStep 5: Model training and evaluation complete for basic models.")


# --- Step 6: Integrate models into a system (Conceptual) ---
print("\n--- Step 6: Integrate models into a system (Conceptual) ---")
print("1. Model Deployment Strategy: Serialization, containerization, orchestration.")
print("2. New Data Integration: Batch and real-time processing, data pipelines, validation.")
print("3. Serving Model Predictions: REST APIs, dashboards, automated reports, alerts.")
print("4. Presentation and Utilization of Model Output: Tailored output for each domain.")
print("5. Conceptual Architecture and Data Flow: Data sources, ingestion, storage, processing, training, registry, prediction service, output, feedback loop.")
print("-" * 50)


# --- Step 7: Monitor and maintain the system (Conceptual) ---
print("\n--- Step 7: Monitor and maintain the system (Conceptual) ---")
print("1. Define Key Performance Indicators (KPIs): MSE, R2, Accuracy, Precision, Recall, F1-score, ROC-AUC for relevant tasks.")
print("2. Establish a monitoring framework: Data collection, prediction/outcome logging, performance calculation, dashboards, triggers.")
print("3. Set up alerts and notifications: Thresholds, alerting mechanism, content, escalation policy.")
print("4. Implement feedback collection: Integrated UI, forms, meetings, logging, analysis.")
print("5. Define retraining strategy and triggers: Performance degradation, data/concept drift, schedule, new data, business changes.")
print("6. Steps for retraining and re-evaluating: Data refresh, preprocessing, splitting, training, tuning, evaluation, comparison.")
print("7. Process for deploying retrained models: Versioning, staging testing, approval, deployment methods (Blue/Green, Canary, Rolling), post-deployment monitoring, rollback, decommissioning.")
print("8. Document monitoring and maintenance procedures: Framework, alerts, strategy, playbook, feedback, roles, scheduled maintenance.")
print("-" * 50)

# --- Step 8: Finish task (Summary) ---
print("\n--- Step 8: Summary ---")
print("\nData Analysis Key Findings:")
print("* Project plan outlined for AI integration in railway procurement across 5 domains.")
print("* Potential data sources and conceptual extraction identified.")
print("* Data cleaning, preprocessing, and feature engineering steps defined.")
print("* Specific prediction tasks and suitable AI models selected.")
print("* Basic model training and evaluation demonstrated with limitations noted.")
print("* Conceptual integration framework outlined for deployment and serving.")
print("* Detailed monitoring and maintenance plan established.")

print("\nInsights or Next Steps:")
print("* Acquire real-world data for practical implementation and rigorous analysis.")
print("* Develop robust data pipelines and experiment with advanced models.")
print("* Establish technical infrastructure for deployment, monitoring, and automated retraining.")
print("-" * 50)