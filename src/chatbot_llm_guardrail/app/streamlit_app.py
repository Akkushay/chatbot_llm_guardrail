# src/chatbot_llm_guardrail/app/streamlit_app.py
import os
import streamlit as st
import json
import pandas as pd
import plotly.express as px
from chatbot_llm_guardrail.evaluation.evaluator import Evaluate

os.environ["STREAMLIT_WATCHED_FILES"] = ""
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

st.set_page_config(layout="wide", page_title="LLM Guardrail Evaluation")

default_response = "Our Gold Credit Card offers a 10% APR and an annual fee of $80, as stated in our official docs."
default_docs = [
    {"doc_id": "DOC1234", "content": "Credit Card offers 10% APR and $80 annual fee."},
    {"doc_id": "DOC5678", "content": "Home Loan requires a 20% down payment."},
    {"doc_id": "DOC9101", "content": "Personal Loan interest rate is 8% per annum."},
    {"doc_id": "DOC1121", "content": "Savings account offers 1.5% interest with no fees."},
    {"doc_id": "DOC3141", "content": "Term Deposit has a 1-year duration with 2% interest."}
]
default_docs_df = pd.DataFrame(default_docs)

tab1, tab2 = st.tabs(["Response Testing", "Bulk Evaluation"])

with tab1:
    st.header("Response Testing")
    st.write("Enter a chatbot response and review the retrieved documents.")
    response_input = st.text_area("Chatbot Response", value=default_response, height=150)
    st.subheader("Retrieved Documents")
    st.dataframe(default_docs_df, width=1800)
    if st.button("Evaluate Single Response"):
        try:
            evaluator = Evaluate()
            result = evaluator.evaluate_one_sample(response_input, default_docs)
            st.subheader("Evaluation Result")
            result_df = pd.DataFrame(list(result.items()), columns=["Metric", "Value"])
            st.table(result_df)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

with tab2:
    st.header("Bulk Evaluation")
    st.write("Evaluate all samples in the synthetic dataset.")
    if st.button("Evaluate All Samples"):
        try:
            evaluator = Evaluate()
            evaluation_df, metrics_dict = evaluator.evaluate_all_samples()
            display_cols = ["Query", "Response", "Grounded", "Output"]
            for col in display_cols:
                if col not in evaluation_df.columns:
                    evaluation_df[col] = ""
            evaluation_df = evaluation_df[display_cols]
            evaluation_df = evaluation_df.astype(str)
            st.subheader("Evaluation Results Table")
            def color_output(val):
                return "background-color: green; color: white;" if val == "Correct" else "background-color: orange; color: white;"
            styled_df = evaluation_df.style.applymap(color_output, subset=["Output"])
            st.dataframe(styled_df, width=1800)
            st.subheader("Overall Evaluation Metrics")
            plot_keys = ["accuracy", "precision", "recall", "f1_score"]
            plot_metrics = {}
            for key in plot_keys:
                try:
                    plot_metrics[key] = float(metrics_dict.get(key, 0))
                except Exception as e:
                    st.error(f"Error converting metric '{key}' to float: {e}")
                    plot_metrics[key] = 0.0
            fig = px.bar(
                x=list(plot_metrics.keys()),
                y=list(plot_metrics.values()),
                color=list(plot_metrics.keys()),
                labels={'x': 'Metric', 'y': 'Score'},
                title="Overall Evaluation Metrics",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Bulk evaluation failed: {e}")
