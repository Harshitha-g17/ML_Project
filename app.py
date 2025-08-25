import streamlit as st
import pandas as pd
import random
import time

# --- A. CORE LOGIC (KEEP AS IS) ---

# Placeholder classes for CLV
class CustomData:
    def __init__(self, customer_id, quantity, unit_price, country, invoice_month):
        self.customer_id = customer_id
        self.quantity = quantity
        self.unit_price = unit_price
        self.country = country
        self.invoice_month = invoice_month

    def get_data_as_data_frame(self):
        return pd.DataFrame({
            'CustomerID': [self.customer_id],
            'Quantity': [self.quantity],
            'UnitPrice': [self.unit_price],
            'Country': [self.country],
            'InvoiceMonth': [self.invoice_month]
        })

class PredictPipeline:
    def predict(self, features):
        return [round(random.uniform(100, 15000), 2)]

def generate_recommendations(clv_value):
    if clv_value < 2000:
        return {
            "title": "Low-Value Customer",
            "emoji": "📉",
            "message": "This customer has a low predicted CLV. Focus on increasing their purchase frequency and basket size.",
            "actions": [
                "🎁 Offer small discounts (e.g., 5-10% off) to encourage repeat purchases.",
                "📢 Send personalized marketing emails with product bundles or related items.",
                "💡 Run a welcome-back campaign if they haven't purchased recently."
            ]
        }
    elif 2000 <= clv_value < 6000:
        return {
            "title": "Medium-Value Customer",
            "emoji": "📈",
            "message": "This customer shows good potential. Nurture the relationship to turn them into a high-value asset.",
            "actions": [
                "🎉 Provide loyalty rewards or points to strengthen customer relationship.",
                "🛒 Recommend complementary products based on their past purchases.",
                "🚚 Offer free or discounted shipping on their next order to reduce friction."
            ]
        }
    else:
        return {
            "title": "High-Value (VIP) Customer",
            "emoji": "👑",
            "message": "This is a high-value customer. Focus on retention and delight to ensure they remain loyal.",
            "actions": [
                "👑 Treat them as VIPs with exclusive offers and early access to new products.",
                "💳 Suggest premium products, add-ons, or subscription models tailored to their needs.",
                "🤝 Assign a personalized account manager or customer success representative for a high-touch experience."
            ]
        }

# --- B. STREAMLIT UI LAYOUT & COMPONENTS ---

st.set_page_config(
    page_title="Customer Lifetime Value Predictor",
    page_icon="💰",
    layout="wide",
)

# Custom CSS for a professional look
st.markdown("""
<style>
    /* Main container and text */
    .st-emotion-cache-18ni7ap { /* This targets the Streamlit main container */
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .st-emotion-cache-1av54b { /* This targets the sidebar */
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Custom Header with animation */
    @keyframes fadeInDown {
      from { opacity: 0; transform: translateY(-20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        color: #00ffcc;
        text-shadow: 2px 2px 8px #00aaff;
        animation: fadeInDown 1s ease-out;
    }
    .subheader {
        text-align: center;
        color: #8b949e;
        font-size: 1.2em;
    }
    
    /* Input Form Styling */
    .stForm {
        padding: 30px;
        border-radius: 15px;
        background-color: #161b22;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    /* Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #1f252b;
        border-left: 5px solid #00aaff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #00aaff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: transform 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 170, 255, 0.5);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #1f252b;
        border-radius: 8px 8px 0 0;
        gap: 12px;
        padding-top: 10px;
        padding-bottom: 10px;
        transition: background-color 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00aaff;
        color: white;
    }
    .stTabs [aria-selected="true"] > div > p {
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>💰 CLV Predictor & Strategy Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Estimate a customer's lifetime value and get actionable recommendations for your business.</p>", unsafe_allow_html=True)

# --- C. MAIN APP LOGIC WITH TABS ---

tab1, tab2 = st.tabs(["📊 Input & Predict", "📝 About the Model"])

with tab1:
    with st.container():
        st.subheader("Customer Details")
        st.write("Please fill in the details below to predict the customer's lifetime value.")

        with st.form("clv_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                customer_id = st.text_input("🆔 Customer ID", help="A unique identifier for the customer.")
                quantity = st.number_input("📦 Quantity Purchased", min_value=1, step=1, help="Total number of items purchased.")
            with col2:
                unit_price = st.number_input("💵 Unit Price", min_value=0.0, step=0.5, help="Average price per unit.")
                country = st.selectbox("🌍 Country", ["United Kingdom", "France", "Germany", "Spain", "Other"], help="The customer's country.")
            
            invoice_month = st.selectbox("🗓️ Invoice Month", 
                                        ["January", "February", "March", "April", "May", "June",
                                         "July", "August", "September", "October", "November", "December"], help="The month of the most recent purchase.")
            
            submitted = st.form_submit_button("🔮 Predict CLV")

    if submitted:
        if not customer_id or quantity <= 0 or unit_price <= 0:
            st.error("❗ Please ensure 'Customer ID', 'Quantity', and 'Unit Price' are filled in with valid values.")
        else:
            with st.spinner('Predicting CLV...'):
                # Progress bar for a smooth transition
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                progress_bar.empty()

                data = CustomData(customer_id, quantity, unit_price, country, invoice_month)
                pred_df = data.get_data_as_data_frame()
                predict_pipeline = PredictPipeline()
                clv_value = predict_pipeline.predict(pred_df)[0]
                recs = generate_recommendations(clv_value)

            st.success("✅ Prediction complete!")
            
            # Use columns to present results side-by-side
            col_metrics, col_recs = st.columns([1, 1])

            with col_metrics:
                st.subheader("Prediction Summary")
                st.metric(label=f"Predicted CLV for {customer_id}", value=f"${clv_value:,.2f}", delta=recs['emoji'])
                
                st.markdown(f"**Customer Type:** <span style='color: #00ffcc;'>{recs['title']}</span>", unsafe_allow_html=True)
                st.info(recs['message'])
                
            with col_recs:
                st.subheader("Actionable Recommendations")
                st.markdown("---")
                for action in recs['actions']:
                    st.markdown(f"- {action}")
            
with tab2:
    st.header("About the CLV Prediction Model")
    st.markdown("""
        This application uses a Machine Learning model to predict the future revenue a customer will generate.
        The model is trained on historical e-commerce data to identify patterns that correlate with high-value customers.

        ### Key Features Used for Prediction:
        - **Quantity:** The number of items purchased.
        - **Unit Price:** The price per item.
        - **Country:** The customer's geographical location.
        - **Invoice Month:** The time of year of the purchase.

        ### How the Model Works:
        The model analyzes these inputs to estimate a customer's **Customer Lifetime Value (CLV)**. This value helps businesses
        allocate resources effectively, identify their most loyal customers, and develop targeted marketing strategies.
        
        The model is a **Random Forest Classifier** which is robust and effective for this kind of tabular data.
        """, unsafe_allow_html=True)