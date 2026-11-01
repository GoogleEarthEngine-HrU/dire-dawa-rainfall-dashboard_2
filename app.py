import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dire Dawa Extreme Rainfall Dashboard", layout="wide"
)

st.title("Dire Dawa Extreme Rainfall and Drought Analysis")
st.write(
    "Interactive dashboard displaying GEV and POT extreme precipitation models"
    " (1981-2025)."
)

output_dir = "revised_analysis_final"

# Load Validation Report
report_path = os.path.join(output_dir, "VALIDATION_AND_METHODS_REPORT.json")
if os.path.exists(report_path):
  with open(report_path, "r") as f:
    report = json.load(f)

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Daily Records", report["validation"]["daily_records"])
  col2.metric(
      "Mean Annual Max (mm)", round(report["mean_annual_maximum_mm"], 2)
  )
  col3.metric("POT Threshold (mm)", round(report["pot_threshold_mm"], 2))
  col4.metric("Independent Events", report["pot_independent_events"])

# Load Return Level Comparison
comparison_path = os.path.join(output_dir, "13_GEV_POT_Comparison_REVISED.csv")
if os.path.exists(comparison_path):
  st.subheader("GEV vs POT Return Level Comparison")
  df_comp = pd.read_csv(comparison_path)
  st.dataframe(df_comp, width="stretch")

  st.bar_chart(
      df_comp.set_index("Return_Period_years")[
          ["GEV_Return_Level_mm", "POT_Return_Level_mm"]
      ]
  )

# Display Generated Images
st.subheader("Diagnostic Plots")
img_files = [
    "01_Daily_Rainfall_REVISED.png",
    "02_Annual_Maximum_Rainfall_REVISED.png",
    "05_GEV_Distribution_Fit_REVISED.png",
    "14_GEV_POT_Comparison_REVISED.png",
]

selected_img = st.selectbox("Select Diagnostic Plot", img_files)
img_path = os.path.join(output_dir, selected_img)
if os.path.exists(img_path):
  st.image(img_path, caption=selected_img, width="stretch")
