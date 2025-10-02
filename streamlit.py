import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import io

df = pd.read_csv("updated_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df['DayOfWeek'] = df['Date'].dt.day_name()
df['Hour'] = pd.to_datetime(df['Time']).dt.hour
df['Month'] = df['Date'].dt.to_period('M')

st.set_page_config(page_title="UPI Financial Analyzer", layout="wide", page_icon="💰")
st.title("💰 Personal UPI Usage & Financial Analyzer")
st.write("Analyze your UPI transactions with AI-powered insights.")

start_date = st.date_input("Start Date", df['Date'].min())
end_date = st.date_input("End Date", df['Date'].max())
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

categories = df_filtered['Category'].unique()
selected_categories = st.multiselect("Select Categories", categories, default=categories)
df_filtered = df_filtered[df_filtered['Category'].isin(selected_categories)]

category_summary = df_filtered.groupby("Category")["Amount"].sum().sort_values(ascending=False)
day_summary = df_filtered.groupby("DayOfWeek")["Amount"].sum().sort_values(ascending=False)
hour_summary = df_filtered.groupby("Hour")["Amount"].sum().sort_values(ascending=False)
top_merchants = df_filtered.groupby("Receiver")["Amount"].sum().sort_values(ascending=False).head(5)

small_transactions = df_filtered[df_filtered['Amount'] < 500]
recurring = small_transactions.groupby(['Receiver', 'Month']).size()
wasteful_small = recurring[recurring > 3]
wasteful_status = df_filtered[df_filtered['Status'].isin(['PENDING', 'FAILED'])]

API_KEY = ""
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

llm_prompt = f"""
You are a financial advisor AI. Analyze the user's UPI data and provide clear, actionable insights.

Spending patterns by category: {category_summary.to_dict()}
Spending by day: {day_summary.to_dict()}
Spending by hour: {hour_summary.to_dict()}
Top merchants: {top_merchants.to_dict()}
Small & recurring transactions: {wasteful_small.to_dict()}
Pending/Failed transactions: {wasteful_status.to_dict(orient='records')}
"""

response = model.generate_content(llm_prompt)
advice_text = response.text if response and response.text else "No advice generated."

def display_category_spending():
    st.header("📊 Category-wise Spending")
    fig, ax = plt.subplots()
    category_summary.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("🥧 Category Distribution (Pie Chart)")
    fig_pie, ax_pie = plt.subplots()
    category_summary.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax_pie)
    ax_pie.set_ylabel("")
    st.pyplot(fig_pie)

def display_day_hour_analysis():
    st.header("🗓️ Spending by Day")
    fig1, ax1 = plt.subplots()
    day_summary.plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

    st.header("⏰ Spending by Hour")
    fig2, ax2 = plt.subplots()
    hour_summary.plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

def display_top_merchants():
    st.header("🏪 Top 5 Merchants")
    st.table(top_merchants)

def display_monthly_trend():
    st.header("📈 Monthly Spending Trend")
    monthly_trend = df_filtered.groupby(df_filtered['Date'].dt.to_period('M'))['Amount'].sum()
    monthly_trend.index = monthly_trend.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_trend.plot(kind='line', marker='o', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def display_wasteful_transactions():
    st.header("⚠️ Potential Wasteful Transactions")
    if not wasteful_small.empty:
        st.write("**Small repetitive transactions detected:**")
        st.write(wasteful_small)
    else:
        st.success("No repetitive small transactions ✅")

    if not wasteful_status.empty:
        st.write("**Pending/Failed transactions:**")
        st.dataframe(wasteful_status[['Date','Receiver','Amount','Status']])
    else:
        st.success("No pending or failed transactions ✅")

def display_ai_advice():
    st.header("🤖 AI-Powered Financial Advice")
    st.info(advice_text)

def export_pdf():
    st.header("📄 Export Report as PDF")
    if st.button("Download Full PDF Report"):
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for plot_func in [display_category_spending, display_day_hour_analysis, display_monthly_trend]:
                fig = plt.figure()
                plot_func()
                plt.close(fig)
                fig.savefig(pdf, format='pdf')

        pdf_text = FPDF()
        pdf_text.add_page()
        pdf_text.set_font("Arial", size=12)
        pdf_text.multi_cell(0, 8, txt="🤖 AI-Powered Financial Advice:\n\n" + advice_text)
        pdf_text_buffer = io.BytesIO()
        pdf_text.output(pdf_text_buffer)
        pdf_text_buffer.seek(0)

        st.download_button("Download Charts PDF", data=pdf_buffer, file_name="UPI_Charts_Report.pdf", mime="application/pdf")
        st.download_button("Download AI Advice PDF", data=pdf_text_buffer, file_name="UPI_AI_Advice.pdf", mime="application/pdf")
        st.success("✅ PDFs are ready for download!")

menu = [
    "Category Spending",
    "Day & Hour Analysis",
    "Top Merchants",
    "Monthly Trend",
    "Wasteful Transactions",
    "AI Advice",
    "Export Report"
]

selection = st.sidebar.radio("📌 Navigate", menu)

if selection == "Category Spending":
    display_category_spending()
elif selection == "Day & Hour Analysis":
    display_day_hour_analysis()
elif selection == "Top Merchants":
    display_top_merchants()
elif selection == "Monthly Trend":
    display_monthly_trend()
elif selection == "Wasteful Transactions":
    display_wasteful_transactions()
elif selection == "AI Advice":
    display_ai_advice()
elif selection == "Export Report":
    export_pdf()
