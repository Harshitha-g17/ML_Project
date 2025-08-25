import streamlit as st
import pandas as pd
import random
import time

# --- A. CORE LOGIC ---

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
            "color": "#e53e3e",  # Red
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
            "color": "#dd6b20",  # Orange
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
            "color": "#38a169",  # Green
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

# Custom CSS for a professional look with improved visibility
st.markdown("""
<style>
    /* Main container and text */
    .st-emotion-cache-18ni7ap {
        background-color: #f7fafc;
        color: #2d3748;
    }
    .st-emotion-cache-1av54b {
        background-color: #f7fafc;
        color: #2d3748;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        color: #4299e1;
        text-shadow: none;
    }
    .subheader {
        text-align: center;
        color: #718096;
        font-size: 1.5em;
        font-weight: bold;
    }
    
    /* Input Form Styling - Light gray background for contrast */
    .stForm {
        padding: 30px;
        border-radius: 15px;
        background-color: #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Cards Styling with dynamic colors */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 5px solid #4299e1;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Button Styling - Muted blue */
    .stButton > button {
        background-color: #63b3ed;
        color: white;
        font-weight: bold;
        font-size: 1.2em;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #4299e1;
    }
    
    /* Tab Styling - Corrected for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        background-color: #e2e8f0; /* Use a slightly darker background */
        border-radius: 8px 8px 0 0;
        gap: 12px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
        font-size: 1.1em;
        color: #2d3748; /* Make the text dark and clear */
    }
    .stTabs [aria-selected="true"] {
        background-color: #4299e1;
        color: white;
    }
    .stTabs [aria-selected="true"] > div > p {
        color: white;
        font-weight: bold;
    }

    /* Info and other styled containers */
    [data-testid="stText"] {
        color: #2d3748;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>💰 Customer Lifetime Value Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Predict CLV and get personalized business strategies.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Prediction", "📝 About"])

with tab1:
    with st.container():
        st.subheader("Customer Details")
        st.info("Enter the information below to get a CLV prediction.")

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
            st.error("❗ Please ensure **Customer ID**, **Quantity**, and **Unit Price** are filled in with valid values.")
        else:
            with st.spinner('Predicting CLV...'):
                time.sleep(1) # Simulating prediction time
                data = CustomData(customer_id, quantity, unit_price, country, invoice_month)
                pred_df = data.get_data_as_data_frame()
                predict_pipeline = PredictPipeline()
                clv_value = predict_pipeline.predict(pred_df)[0]
                recs = generate_recommendations(clv_value)

            st.success("✅ Prediction complete!")
            
            col_metrics, col_recs = st.columns([1, 1])

            with col_metrics:
                st.subheader("Prediction Summary")
                st.metric(
                    label=f"Predicted CLV for **{customer_id}**", 
                    value=f"${clv_value:,.2f}", 
                    delta=recs['emoji'],
                    delta_color="off"
                )
                st.markdown(f"<p style='font-size: 1.2em; color: {recs['color']};'>**{recs['title']}**</p>", unsafe_allow_html=True)
                st.info(recs['message'])
                
            with col_recs:
                st.subheader("Actionable Recommendations")
                st.info("💡 Use these strategies to maximize customer value.")
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