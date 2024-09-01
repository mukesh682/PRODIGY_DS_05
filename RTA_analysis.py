import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Load dataset
df = pd.read_csv('cleaned.csv')

# Data Cleaning
df = df.dropna()  # Drop rows with missing values

# Define categorical features and ordinal features with natural ordering
categorical_features = [
    'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation',
    'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 'Light_conditions',
    'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 'Pedestrian_movement',
    'Cause_of_accident', 'Driving_experience'
]

ordinal_features = {
    'Driving_experience': ['0-1yrs', '1-2yrs', '3-4yrs', '5-6yrs', '7-8yrs', '9+yrs']
}

# Encode ordinal features
ordinal_encoders = {}
for feature, categories in ordinal_features.items():
    oe = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
    df[feature] = oe.fit_transform(df[[feature]].astype(str))
    ordinal_encoders[feature] = oe

# Encode other categorical features using LabelEncoder
label_encoders = {}
for feature in categorical_features:
    if feature not in ordinal_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))  # Convert to string to avoid issues
        label_encoders[feature] = le

# Define features and target
features = df.drop('Accident_severity', axis=1)
target = df['Accident_severity']

# Function to map encoded values back to original categories
def map_encoded_to_original(data, feature, encoder):
    return encoder.inverse_transform(data[feature])

# Univariate Analysis
def plot_categorical_distribution(data, feature, encoder=None):
    if encoder:
        feature_values = map_encoded_to_original(data, feature, encoder)
    else:
        feature_values = data[feature]
    
    fig = px.histogram(data, x=feature_values, title=f'Distribution of {feature}')
    fig.update_layout(xaxis_title=feature, yaxis_title='Count')
    fig.show()

for feature in categorical_features:
    encoder = label_encoders.get(feature) if feature not in ordinal_features else None
    plot_categorical_distribution(df, feature, encoder)

# Bivariate Analysis
def plot_bivariate_distribution(data, x_feature, hue_feature):
    fig = px.histogram(data, x=x_feature, color=hue_feature, barmode='group',
                       title=f'{hue_feature} by {x_feature}')
    fig.update_layout(xaxis_title=x_feature, yaxis_title='Count')
    fig.show()

plot_bivariate_distribution(df, 'Weather_conditions', 'Accident_severity')
plot_bivariate_distribution(df, 'Road_surface_type', 'Accident_severity')

# Predictive Modeling
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(f'{model.__class__.__name__}:')
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(f'Accuracy: {accuracy_score(y_test, predictions):.2f}')
    
    # Handle multi-class ROC-AUC calculation
    if hasattr(model, 'predict_proba'):
        try:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            print(f'ROC-AUC: {roc_auc:.2f}')
        except ValueError as e:
            print(f'ROC-AUC calculation error: {e}')

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train and evaluate models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

# Feature Importance
def plot_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = features.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', title='Feature Importance',
                    labels={'Importance': 'Importance', 'Feature': 'Feature'})
        fig.update_layout(yaxis_title='Feature', xaxis_title='Importance')
        fig.show()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
plot_feature_importance(rf_model, features)

# Segmentation Analysis
def plot_segmentation(data, x_feature, hue_feature):
    fig = px.histogram(data, x=x_feature, color=hue_feature, barmode='group',
                       title=f'{hue_feature} by {x_feature}')
    fig.update_layout(xaxis_title=x_feature, yaxis_title='Count')
    fig.show()

plot_segmentation(df, 'Age_band_of_driver', 'Accident_severity')
plot_segmentation(df, 'Sex_of_driver', 'Accident_severity')
plot_segmentation(df, 'Cause_of_accident', 'Accident_severity')
plot_segmentation(df, 'Vehicle_movement', 'Accident_severity')
