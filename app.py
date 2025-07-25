import streamlit as st
import pytesseract
from PIL import Image
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import re
import plotly.express as px
import time
import seaborn as sns
import matplotlib.pyplot as plt

# MUST be first Streamlit command
st.set_page_config(
    page_title="Small Business Loan Predictor", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### A smart loan prediction system for small businesses\nDeveloped by Mehluli Nokwara"
    }
)

# Custom CSS for animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .animated {
        animation: fadeIn 0.5s ease-out;
    }
    .slide-animation {
        animation: slideIn 0.7s ease-out;
    }
    .document-preview {
        text-align: center;
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .reject-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #dc3545;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .verification-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #ffc107;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset with company info
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('loan_data.csv')  # Updated CSV
        # Ensure required columns exist
        required_cols = ['company_name', 'CEO', 'years_in_business', 'monthly_revenue',
                        'existing_loans', 'loan_amount_requested', 'collateral_value',
                        'approved_status', 'approved_amount']
        if not all(col in df.columns for col in required_cols):
            st.error("CSV is missing required columns")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

df = load_data()

# Train models and evaluate them
@st.cache_resource
def train_and_evaluate_models():
    X = df[['years_in_business', 'monthly_revenue', 'existing_loans', 
            'loan_amount_requested', 'collateral_value']]
    y_approved = df['approved_status']
    y_amount = df['approved_amount']

    # Split data for approval model
    X_train, X_test, y_train_approved, y_test_approved = train_test_split(
        X, y_approved, test_size=0.2, random_state=42, stratify=y_approved)
    
    # Split data for amount model (only approved loans)
    approved_mask = y_approved == 1
    X_train_amount, X_test_amount, y_train_amount, y_test_amount = train_test_split(
        X[approved_mask], y_amount[approved_mask], test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Train approval model (Logistic Regression)
    model_approved = LogisticRegression(max_iter=1000)
    model_approved.fit(X_train, y_train_approved)
    
    # Evaluate approval model
    y_pred_approved = model_approved.predict(X_test)
    y_prob_approved = model_approved.predict_proba(X_test)[:, 1]
    
    logreg_metrics = {
        'accuracy': accuracy_score(y_test_approved, y_pred_approved),
        'precision': precision_score(y_test_approved, y_pred_approved),
        'recall': recall_score(y_test_approved, y_pred_approved),
        'f1': f1_score(y_test_approved, y_pred_approved),
        'roc_auc': roc_auc_score(y_test_approved, y_prob_approved),
        'confusion_matrix': confusion_matrix(y_test_approved, y_pred_approved)
    }

    # Train amount model (Linear Regression)
    model_amount = LinearRegression()
    model_amount.fit(X_train_amount[numerical_cols], y_train_amount)
    
    # Evaluate amount model
    y_pred_amount = model_amount.predict(X_test_amount[numerical_cols])
    amount_metrics = {
        'r2_score': model_amount.score(X_test_amount[numerical_cols], y_test_amount),
        'mae': np.mean(np.abs(y_pred_amount - y_test_amount)),
        'mse': np.mean((y_pred_amount - y_test_amount)**2),
        'rmse': np.sqrt(np.mean((y_pred_amount - y_test_amount)**2))
    }

    return model_approved, model_amount, scaler, numerical_cols, logreg_metrics, amount_metrics

model_approved, model_amount, scaler, numerical_cols, logreg_metrics, amount_metrics = train_and_evaluate_models()

# Text extraction functions
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
        'company_name': r'(?:Company|Business|Enterprise)\s*Name[:\s]*([^\n]+)',
        'ceo_name': r'(?:CEO|Owner|Director|Proprietor)[:\s]*([^\n]+)',
        'years_in_business': r'(?:Years\D*business|YIB|Operating)[:\s]*(\d+)',
        'monthly_revenue': r'(?:Monthly\D*revenue|Revenue|Turnover)[:\s]*(\d+)',
        'existing_loans': r'(?:Existing\D*loans|Current\D*loans|Debts)[:\s]*(\d+)',
        'loan_amount_requested': r'(?:Loan\D*amount|Amount\D*requested|Funding\D*needed)[:\s]*(\d+)',
        'collateral_value': r'(?:Collateral\D*value|Asset\D*value|Security\D*value)[:\s]*(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        info[key] = match.group(1).strip() if match else None
    
    return info

def verify_company(info, df):
    """Check if company name matches our records"""
    if not info['company_name']:
        return False, "Missing company name information"
        
    # Case-insensitive comparison with strip
    company_matches = df['company_name'].str.lower().str.strip() == info['company_name'].lower().strip()
    
    matching_records = df[company_matches]
    
    if matching_records.empty:
        return False, "Company not found in our records"
    return True, "Verification successful"

def predict_loan_approval(info, input_data):
    """Predict loan approval after verification"""
    try:
        input_data_scaled = input_data.copy()
        input_data_scaled[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        prediction_approved = model_approved.predict(input_data_scaled)
        if prediction_approved[0] == 1:
            prediction_amount = max(0, model_amount.predict(input_data_scaled[numerical_cols])[0])
            approved_amount = min(prediction_amount, input_data['loan_amount_requested'].iloc[0])
            return prediction_approved[0], approved_amount
        return prediction_approved[0], 0
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# UI Components
st.markdown('<div class="slide-animation"><h1 style="text-align:center; color: #2c3e50;">Small Business Loan Predictor</h1></div>', 
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Verification Requirements")
    st.markdown("""
    <div class="metric-card">
        ✔ Valid registered business<br>
        ✔ Matching company name<br>
        ✔ Complete financial details
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Approval Factors")
    st.markdown("""
    <div class="metric-card">
        <strong>Positive Factors:</strong><br>
        ✓ Business longevity<br>
        ✓ Monthly revenue<br>
        ✓ Collateral coverage<br><br>
        <strong>Negative Factors:</strong><br>
        ✗ Existing debt burden<br>
        ✗ Low credit score
    </div>
    """, unsafe_allow_html=True)

# Main App
tab1, tab2 = st.tabs(["📄 Document Processing", "📊 Business Insights"])

with tab1:
    st.markdown("### Upload Business Documents")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse files", 
        type=['pdf', 'jpg', 'jpeg', 'png'],
        help="Supported formats: PDF, JPG, PNG",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        with st.spinner("Analyzing your documents..."):
            text = extract_text_from_file(uploaded_file)
            info = extract_info(text)
            
            if all(info.values()):
                # Display extracted info
                with st.expander("📋 Extracted Business Details", expanded=True):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Company Name</h4>
                            <h3>{info['company_name']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>CEO/Owner</h4>
                            <h3>{info['ceo_name']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Years Operating</h4>
                            <h3>{info['years_in_business']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Monthly Revenue</h4>
                            <h3>${info['monthly_revenue']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Existing Loans</h4>
                            <h3>{info['existing_loans']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Loan Requested</h4>
                            <h3>${info['loan_amount_requested']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Collateral Value</h4>
                            <h3>${info['collateral_value']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Verify business
                is_verified, verification_msg = verify_company(info, df)
                
                if not is_verified:
                    st.markdown(f"""
                    <div class="verification-box">
                        <h3>⚠️ Verification Failed</h3>
                        <p>{verification_msg}</p>
                        <p>Please ensure:</p>
                        <ul>
                            <li>Business is properly registered</li>
                            <li>Company name matches official records</li>
                            <li>Name is spelled correctly</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ Business Verified")
                    time.sleep(0.5)
                    
                    # Prepare data for prediction
                    input_data = pd.DataFrame({
                        'years_in_business': [int(info['years_in_business'])],
                        'monthly_revenue': [int(info['monthly_revenue'])],
                        'existing_loans': [int(info['existing_loans'])],
                        'loan_amount_requested': [int(info['loan_amount_requested'])],
                        'collateral_value': [int(info['collateral_value'])]
                    })
                    
                    with st.spinner("Calculating loan eligibility..."):
                        approved, amount = predict_loan_approval(info, input_data)
                        
                        if approved == 1:
                            st.markdown(f"""
                            <div class="success-box">
                                <h3>🎉 Loan Approved!</h3>
                                <p><strong>Approved Amount:</strong> ${amount:,.2f}</p>
                                <p>This represents {amount/int(info['loan_amount_requested'])*100:.1f}% of requested amount</p>
                                <p>Next steps: Contact our office to complete paperwork</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        elif approved == 0:
                            st.markdown(f"""
                            <div class="reject-box">
                                <h3>❌ Loan Not Approved</h3>
                                <p>Based on our assessment, your business doesn't meet current criteria.</p>
                                <p>Recommendations:</p>
                                <ul>
                                    <li>Increase revenue or collateral</li>
                                    <li>Reduce existing debt</li>
                                    <li>Reapply after 6 months</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("Incomplete application document. Missing:")
                missing = [k.replace('_', ' ').title() for k, v in info.items() if not v]
                st.write(", ".join(missing))

with tab2:
    st.subheader("📈 Business Loan Insights Dashboard")
    
    # EDA Section
    st.markdown("### 🔍 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Data Distribution")
        selected_col = st.selectbox("Select column to analyze", 
                                  ['years_in_business', 'monthly_revenue', 
                                   'existing_loans', 'loan_amount_requested', 
                                   'collateral_value'])
        fig = px.histogram(df, x=selected_col, nbins=30, 
                          title=f'Distribution of {selected_col.replace("_", " ").title()}',
                          color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("#### Outlier Analysis")
        fig = px.box(df, y=selected_col, 
                    title=f'Box Plot of {selected_col.replace("_", " ").title()}',
                    color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Correlation Analysis")
    # FIXED: Proper list concatenation for correlation matrix
    corr_cols = numerical_cols.copy()
    corr_cols.extend(['approved_status', 'approved_amount'])
    corr_matrix = df[corr_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                   text_auto=True, 
                   aspect="auto",
                   title="Feature Correlation Matrix",
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Skewness Analysis
    st.markdown("#### Skewness Analysis")
    skewness = df[numerical_cols].skew().reset_index()
    skewness.columns = ['Feature', 'Skewness']
    fig = px.bar(skewness, x='Feature', y='Skewness', 
                title='Feature Skewness (0 = symmetric)',
                color='Skewness',
                color_continuous_scale='RdYlGn',
                range_color=[-2, 2])
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Evaluation Section
    st.markdown("### � Model Evaluation Metrics")
    
    st.markdown("#### 📊 Logistic Regression (Approval Prediction)")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{logreg_metrics['accuracy']:.2%}")
    cols[1].metric("Precision", f"{logreg_metrics['precision']:.2%}")
    cols[2].metric("Recall", f"{logreg_metrics['recall']:.2%}")
    cols[3].metric("F1 Score", f"{logreg_metrics['f1']:.2%}")
    cols[4].metric("ROC AUC", f"{logreg_metrics['roc_auc']:.2%}")
    
    st.markdown("##### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(logreg_metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Blues', ax=ax,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    st.markdown("#### 📈 Linear Regression (Amount Prediction)")
    cols = st.columns(4)
    cols[0].metric("R² Score", f"{amount_metrics['r2_score']:.2%}")
    cols[1].metric("MAE", f"${amount_metrics['mae']:,.2f}")
    cols[2].metric("MSE", f"${amount_metrics['mse']:,.2f}")
    cols[3].metric("RMSE", f"${amount_metrics['rmse']:,.2f}")
    
    # Original visualization code remains the same
    st.markdown("---")
    st.subheader("Original Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names='approved_status', 
                    title='Approval Rate Distribution', 
                    labels={'0': 'Rejected', '1': 'Approved'},
                    color_discrete_sequence=['#dc3545', '#28a745'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        approved_df = df[df['approved_status'] == 1]
        fig = px.histogram(approved_df, x='approved_amount',
                          title='Approved Loan Amount Distribution',
                          labels={'approved_amount': 'Amount ($)'},
                          color_discrete_sequence=['#28a745'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    fig = px.scatter(df, x='monthly_revenue', y='approved_amount',
                    color='approved_status',
                    title='Revenue vs Approved Amount',
                    labels={'monthly_revenue': 'Monthly Revenue ($)'},
                    color_discrete_sequence=['#dc3545', '#28a745'])
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 14px; color: #666; margin-top: 50px;">
    <p>Developed with ❤️ by Mehluli Nokwara • © 2023 Small Business Loan Assessment System</p>
</div>
""", unsafe_allow_html=True)
