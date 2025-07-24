import streamlit as st
import pytesseract
from PIL import Image
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import re
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Loan Approval Predictor", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #2b5876 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px !important;
        color: #4e4376 !important;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #4e4376 !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #2b5876 !important;
        color: white !important;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .reject-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    .feature-box {
        background-color: #f0f5ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .document-preview {
        text-align: center;
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('loan_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

df = load_data()

# Train models
@st.cache_resource
def train_models():
    X = df[['years_in_business', 'applicant_age', 'monthly_salary', 'existing_loans', 'loan_amount_requested', 'collateral_value']]
    y_approved = df['approved_status']
    y_amount = df['approved_amount']

    X_train, X_test, y_train_approved, y_test_approved = train_test_split(X, y_approved, test_size=0.2, random_state=42, stratify=y_approved)
    approved_mask = y_approved == 1
    X_train_amount, X_test_amount, y_train_amount, y_test_amount = train_test_split(X[approved_mask], y_amount[approved_mask], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    model_approved = LogisticRegression(max_iter=1000)
    model_approved.fit(X_train, y_train_approved)

    model_amount = LinearRegression()
    model_amount.fit(X_train_amount[numerical_cols], y_train_amount)

    return model_approved, model_amount, scaler, numerical_cols

model_approved, model_amount, scaler, numerical_cols = train_models()

# App header
st.markdown('<div class="header">Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown("""
Welcome to our AI-powered loan approval system. Upload your loan application document (PDF or image) 
to get an instant prediction about your loan approval status and potential approved amount.
""")

# Sidebar with stats
with st.sidebar:
    st.markdown("### 📊 Dataset Statistics")
    st.write(f"Total applications: {len(df):,}")
    st.write(f"Approval rate: {df['approved_status'].mean()*100:.1f}%")
    st.write(f"Average approved amount: ${df[df['approved_status']==1]['approved_amount'].mean():,.2f}")
    
    st.markdown("### 📈 Approval Factors")
    st.write("✔️ Years in business")
    st.write("✔️ Monthly salary")
    st.write("✔️ Collateral value")
    st.write("✖️ Existing loans")
    st.write("✖️ Young applicant age")
    
    st.markdown("### ℹ️ How It Works")
    st.write("1. Upload your loan document")
    st.write("2. System extracts key information")
    st.write("3. AI models predict approval")
    st.write("4. Get instant results")

# File processing functions
def extract_text_from_file(file):
    try:
        if file.type == 'application/pdf':
            text = ''
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ''
            return text
        else:
            return pytesseract.image_to_string(Image.open(file))
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return ""

def extract_info(text):
    info = {}
    patterns = {
        'years_in_business': r'(?:Years\D*business|YIB)[:\s]*(\d+)',
        'applicant_age': r'(?:Age|Applicant\D*Age)[:\s]*(\d+)',
        'monthly_salary': r'(?:Monthly\D*salary|Salary)[:\s]*(\d+)',
        'existing_loans': r'(?:Existing\D*loans|Current\D*loans)[:\s]*(\d+)',
        'loan_amount_requested': r'(?:Loan\D*amount|Amount\D*requested)[:\s]*(\d+)',
        'collateral_value': r'(?:Collateral\D*value|Asset\D*value)[:\s]*(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        info[key] = match if match else None
    
    return info

def predict_loan_approval(info):
    try:
        input_data = pd.DataFrame({
            'years_in_business': [int(info['years_in_business'].group(1)) if info['years_in_business'] else 3],
            'applicant_age': [int(info['applicant_age'].group(1)) if info['applicant_age'] else 35],
            'monthly_salary': [int(info['monthly_salary'].group(1)) if info['monthly_salary'] else 5000],
            'existing_loans': [int(info['existing_loans'].group(1)) if info['existing_loans'] else 1],
            'loan_amount_requested': [int(info['loan_amount_requested'].group(1)) if info['loan_amount_requested'] else 20000],
            'collateral_value': [int(info['collateral_value'].group(1)) if info['collateral_value'] else 10000]
        })

        input_data_scaled = input_data.copy()
        input_data_scaled[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        prediction_approved = model_approved.predict(input_data_scaled)
        if prediction_approved[0] == 1:
            prediction_amount = max(0, model_amount.predict(input_data_scaled[numerical_cols])[0])
            approved_amount = min(prediction_amount, input_data['loan_amount_requested'].iloc[0])
            return prediction_approved[0], approved_amount, input_data.iloc[0].to_dict()
        return prediction_approved[0], 0, input_data.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# Main app
tab1, tab2 = st.tabs(["📄 Document Upload", "📊 Data Insights"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload your loan application document", 
        type=['pdf', 'jpg', 'jpeg', 'png'],
        help="Supported formats: PDF, JPG, PNG"
    )

    if uploaded_file is not None:
        with st.spinner("Analyzing your document..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Extracted Information")
                text = extract_text_from_file(uploaded_file)
                info = extract_info(text)
                
                if any(info.values()):
                    features = {
                        'Years in Business': info['years_in_business'].group(1) if info['years_in_business'] else "Not found",
                        'Applicant Age': info['applicant_age'].group(1) if info['applicant_age'] else "Not found",
                        'Monthly Salary': f"${int(info['monthly_salary'].group(1)):,}" if info['monthly_salary'] else "Not found",
                        'Existing Loans': info['existing_loans'].group(1) if info['existing_loans'] else "Not found",
                        'Loan Amount Requested': f"${int(info['loan_amount_requested'].group(1)):,}" if info['loan_amount_requested'] else "Not found",
                        'Collateral Value': f"${int(info['collateral_value'].group(1)):,}" if info['collateral_value'] else "Not found"
                    }
                    
                    for feature, value in features.items():
                        with st.container():
                            st.markdown(f'<div class="feature-box"><strong>{feature}:</strong> {value}</div>', unsafe_allow_html=True)
                    
                    approved, amount, input_data = predict_loan_approval(info)
                    
                    if approved is not None:
                        if approved == 1:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>🎉 Congratulations!</h3>
                                <p>Your loan application has been <strong>approved</strong>!</p>
                                <p><strong>Approved Amount:</strong> ${amount:,.2f}</p>
                                <p>This represents {amount/input_data['loan_amount_requested']*100:.1f}% of your requested amount.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="reject-box">
                                <h3>⚠️ Application Not Approved</h3>
                                <p>Based on our assessment, your application doesn't meet our current criteria.</p>
                                <p>Common reasons include insufficient collateral, high existing debt, or limited business history.</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No loan information could be extracted from the document")

            with col2:
                st.subheader("Document Preview")
                st.markdown(f"""
                <div class="document-preview">
                    <span style="font-size:48px;">{'📄' if uploaded_file.type == 'application/pdf' else '🖼️'}</span>
                    <p>{'PDF Document' if uploaded_file.type == 'application/pdf' else 'Image Document'}</p>
                    <p><small>{uploaded_file.name}</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View extracted text"):
                    st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

with tab2:
    st.subheader("Loan Approval Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df, names='approved_status', 
                    title='Approval Rate Distribution',
                    labels={'0': 'Rejected', '1': 'Approved'})
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        approved_df = df[df['approved_status'] == 1]
        fig = px.histogram(approved_df, x='approved_amount', 
                         title='Approved Amount Distribution',
                         labels={'approved_amount': 'Approved Amount ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Approval Factors")
    
    fig = px.box(df, x='approved_status', y='monthly_salary',
                labels={'approved_status': 'Approval Status', 'monthly_salary': 'Monthly Salary'},
                title='Salary vs Approval Status')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.scatter(df, x='collateral_value', y='loan_amount_requested',
                    color='approved_status',
                    title='Collateral Value vs Requested Amount',
                    labels={'collateral_value': 'Collateral Value ($)',
                           'loan_amount_requested': 'Requested Amount ($)'})
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 14px; color: #666; margin-top: 50px;">
    <p>Developed by Tafadzwa</p>
</div>
""", unsafe_allow_html=True)