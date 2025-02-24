import os
import re
import base64
import streamlit as st
import pandas as pd
import plotly.express as px
from chatbot_llm_guardrail.evaluation.evaluator import Evaluate

###############################################################################
# 1) PAGE CONFIG
###############################################################################
st.set_page_config(
    page_title="LLM Guardrail Evaluation",
    layout="wide"
)

###############################################################################
# 2) CUSTOM CSS FOR A GRAY SIDEBAR AND TABLE STYLING
###############################################################################
st.markdown(
    """
    <style>
    /* Use a global font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  {
      font-family: 'Roboto', sans-serif;
      margin: 0;
      padding: 0;
      background-color: #000000;
      color: #ffffff;
    }

    /* Sidebar container: dark gray background, white text */
    section[data-testid="stSidebar"] > div {
      background-color: #2f2f2f !important;
      color: #fff !important;
      height: 100vh;
      padding: 1rem !important;
    }
    /* Remove default Streamlit spacing around the sidebar heading */
    .css-1d391kg {
      margin: 0 !important;
      padding: 0 !important;
    }

    /* Heading in the sidebar */
    .sidebar-heading {
      font-size: 1.2rem;
      font-weight: 700;
      margin-top: 1rem;
      margin-bottom: 1rem;
    }

    /* Simple search input styling */
    .sidebar-search label {
      color: #fff !important;
      font-weight: 500 !important;
    }
    .sidebar-search input {
      background-color: #444 !important;
      color: #fff !important;
      border-radius: 0.25rem !important;
      border: 1px solid #666 !important;
    }

    /* Radio menu styling (sections) */
    .stRadio > label {
      font-size: 1rem !important;
      font-weight: 500 !important;
      margin-bottom: 0.75rem !important;
      cursor: pointer;
      display: block;
    }
    /* Highlight selected radio option with a lighter background */
    .stRadio > label[aria-checked="true"] {
      background-color: #55555520;
      padding: 0.3rem 0.5rem;
      border-radius: 0.25rem;
    }

    /* 3D table style */
    table.table-3d {
      border-collapse: collapse;
      width: 100%;
      box-shadow: 0 4px 8px rgba(0,0,0,0.15);
      margin-bottom: 1rem;
      background-color: #fff;
    }
    table.table-3d th, table.table-3d td {
      border: 1px solid #ddd;
      padding: 0.5rem;
      text-align: left;
    }

    /* Plot spacing */
    .stPlotlyChart {
      margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

###############################################################################
# 3) EMBED LOCAL IMAGES IN MARKDOWN
###############################################################################
def embed_local_images_in_markdown(md_text: str, base_path=".") -> str:
    """
    Convert local Markdown image references to embedded base64 images.
    """
    pattern = r'!\[(.*?)\]\((.*?)\)'
    matches = re.findall(pattern, md_text)
    if not matches:
        return md_text
    new_md = md_text
    for alt_text, img_path in matches:
        full_path = os.path.join(base_path, img_path)
        if not os.path.exists(full_path):
            continue
        ext = os.path.splitext(full_path)[1].lower()
        mime = "image/png"
        if ext in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif ext == ".gif":
            mime = "image/gif"
        try:
            with open(full_path, "rb") as img_f:
                data = img_f.read()
            b64_data = base64.b64encode(data).decode("utf-8")
            data_uri = f"data:{mime};base64,{b64_data}"
            new_tag = f'<img src="{data_uri}" alt="{alt_text}" width="900"/>'
            old_md = f'![{alt_text}]({img_path})'
            new_md = new_md.replace(old_md, new_tag)
        except Exception as e:
            print(f"Error embedding image {full_path}: {e}")
            continue
    return new_md

###############################################################################
# 4) SIDEBAR CONTENT
###############################################################################
with st.sidebar.container():
    # A search bar at the top (cosmetic example)
    st.markdown('<div class="sidebar-search">', unsafe_allow_html=True)
    search_query = st.text_input("Search...", value="", placeholder="Search here")
    st.markdown("</div>", unsafe_allow_html=True)

    # Heading in the sidebar
    st.markdown('<div class="sidebar-heading">The Bank. - LLM Powered Guardrail</div>', unsafe_allow_html=True)

# Radio-based navigation
navigation = st.sidebar.radio(
    "Navigation",
    ["Single Response", "Bulk Evaluation", "Design Decisions"]
)

###############################################################################
# 5) MAIN CONTENT
###############################################################################
if navigation == "Single Response":
    st.title("Single Response Testing")
    st.write("Enter a chatbot response below and click **Evaluate Single Response** to analyze its groundedness.")
    
    default_response = "Our Gold Credit Card offers a 10% APR and an annual fee of $80, as stated in our official docs."
    default_docs = [
        {"doc_id": "DOC1234", "content": "Credit Card offers 10% APR and $80 annual fee."},
        {"doc_id": "DOC5678", "content": "Home Loan requires a 20% down payment."},
        {"doc_id": "DOC9101", "content": "Personal Loan interest rate is 8% per annum."},
        {"doc_id": "DOC1121", "content": "Savings account offers 1.5% interest with no fees."},
        {"doc_id": "DOC3141", "content": "Term Deposit has a 1-year duration with 2% interest."}
    ]
    df_docs = pd.DataFrame(default_docs)

    st.subheader("Chatbot Response")
    response_input = st.text_area("Chatbot Response:", value=default_response, height=120)

    st.subheader("Retrieved Documents")
    st.table(df_docs.style.set_table_attributes('class="table-3d"'))

    if st.button("Evaluate Single Response"):
        try:
            evaluator = Evaluate()
            result = evaluator.evaluate_one_sample(response_input, default_docs)
            st.write("### Evaluation Result")
            result_df = pd.DataFrame(list(result.items()), columns=["Metric", "Value"])
            st.table(result_df.style.set_table_attributes('class="table-3d"'))
        except Exception as e:
            st.error(f"Error during evaluation: {e}")

elif navigation == "Bulk Evaluation":
    st.title("Bulk Evaluation")
    st.write(
        "Click **Evaluate Bulk Responses** to assess multiple responses at once. "
        # "The 'Query' and 'Response' columns now wrap text and have the same width, "
        # "while 'Grounded' and 'Output' columns remain narrower so they stay on one line."
    )

    if st.button("Evaluate Bulk Responses"):
        try:
            evaluator = Evaluate()
            evaluation_df, metrics_dict = evaluator.evaluate_all_samples()
            
            # Ensure columns exist
            display_cols = ["Query", "Response", "Grounded", "Output"]
            for col in display_cols:
                if col not in evaluation_df.columns:
                    evaluation_df[col] = ""
            evaluation_df = evaluation_df[display_cols].astype(str)

            st.write("### Evaluation Results Table")

            # 1) Wrap text in Query/Response with same width
            # 2) Keep Grounded/Output narrower so text stays in one line
            styled_table = (
                evaluation_df.style
                .set_properties(
                    subset=["Query", "Response"],
                    **{
                        'width': '600px',
                        'white-space': 'normal',
                        'word-wrap': 'break-word',
                        'overflow-wrap': 'break-word'
                    }
                )
                .set_properties(
                    subset=["Grounded", "Output"],
                    **{
                        'width': '100px',
                        'white-space': 'nowrap'
                    }
                )
                .set_table_attributes('class="table-3d"')
            )
            st.table(styled_table)

            st.write("### Overall Evaluation Metrics")
            plot_keys = ["accuracy", "precision", "recall", "f1_score", "METEOR", "ROUGE-L F1", "BERTScore"]
            data_for_chart = []
            for key in plot_keys:
                val = metrics_dict.get(key, 0.0)
                try:
                    val = float(val)
                except:
                    val = 0.0
                data_for_chart.append({"Metric": key, "Score": val})
            df_chart = pd.DataFrame(data_for_chart)
            fig = px.bar(
                df_chart,
                x="Metric",
                y="Score",
                color="Metric",
                title="Overall Evaluation Metrics",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write(
                "**Note:** ROUGE-L F1 may be lower because it depends on exact sequence overlap. "
                "If your chatbot response paraphrases the document, ROUGE might appear low even if the content is correct."
            )
        except Exception as e:
            st.error(f"Bulk evaluation failed: {e}")

else:  # "Design Decisions"
    # st.write("Below is the content of `DESIGN_DECISIONS.md`:")
    try:
        with open("DESIGN_DECISIONS.md", "r", encoding="utf-8") as f:
            raw_md = f.read()
        embedded_md = embed_local_images_in_markdown(raw_md, base_path=".")
        st.markdown(embedded_md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading Design Decisions: {e}")
