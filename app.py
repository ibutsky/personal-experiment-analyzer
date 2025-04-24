import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from io import BytesIO
import base64
from io import StringIO



st.set_page_config(page_title="Run Your Own Experiment", layout="centered")
st.title("🧪 Personal Experiment Analyzer")

st.markdown("""
Use this tool to analyze whether a certain action (like eating a banana before bed) affects an outcome (like sleep hours).
Upload a file, manually enter your data below, or take inspiration from one of the example datasets.""")


st.sidebar.header("📊 Add Some Data")

if "data_source" not in st.session_state:
    st.session_state["data_source"] = "none"
    st.session_state["data"] = None

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("📁 Upload File"):
        st.session_state["data_source"] = "upload"
with col2:
    if st.button("✍️ Manual Entry"):
        st.session_state["data_source"] = "manual"
with col3:
    if st.button("🎓 Sample Dataset"):
        st.session_state["data_source"] = "sample"
        
        

data = pd.read_csv(StringIO("eating a banana,hours of sleep\nyes,7.5\nno,6.0\nyes,8.2\nno,5.5"))
if st.session_state["data_source"] == "upload":
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv", "tsv", "xls",
                                        "xlsx", "pkl", "parquet", "json", "xml"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        try:
            if file_type == 'csv':
                data = pd.read_csv(uploaded_file)
            elif file_type == 'tsv':
                data = pd.read_csv(uploaded_file, sep='\t')
            elif file_type in ['xls', 'xlsx']:
                data = pd.read_excel(uploaded_file)
            elif file_type == "pkl":
                data = pd.read_pickle(uploaded_file)
            elif file_type == "parquet":
                data = pd.read_parquet(uploaded_file)
            elif file_type == "json":
                import json
                raw_json = json.load(uploaded_file)

                # If it's a list of dicts, just use DataFrame directly
                if isinstance(raw_json, list):
                    data = pd.DataFrame(raw_json)
                else:
                    # Try flattening from a likely nested key
                    for key in raw_json:
                        if isinstance(raw_json[key], list):
                            try:
                                data = json_normalize(raw_json[key])
                                break
                            except Exception as e:
                                st.error(f"Could not flatten nested JSON structure: {e}")
                        else:
                            st.error("No list-like structure found in your JSON file.")
            elif file_type == "xml":
                import xml.etree.ElementTree as ET
                tree = ET.parse(uploaded_file)
                root = tree.getroot()

                # Apple Health specific: look for <Record> tags
                records = []
                for record in root.findall('Record'):
                    record_data = record.attrib
                    records.append(record_data)

                data = pd.DataFrame(records)
            else:
                st.error("Unsupported file type.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            
elif st.session_state["data_source"] == "manual":
    #default_df = pd.read_csv(StringIO(default_csv))
    manual_data = st.sidebar.text_area("Manually enter your data as CSV (with headers)",
                           "eating a banana,hours of sleep\nyes,7.5\nno,6.0\nyes,8.2\nno,5.5")
    try:
        from io import StringIO
        data = pd.read_csv(StringIO(manual_data))
    except Exception as e:
        st.error("Failed to parse your input. Check format.")
        data = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)


elif st.session_state["data_source"] == "sample":
    # --- Preloaded samples ---
    sample_datasets = {
    "Sleep Hours vs Energy (Correlation)": "energy_level,hours sleep\n5,6.0\n6,6.5\n7,7.2\n8,7.8\n9,8.1\n10,8.4",
    "Hours Worked vs Productivity (Correlation)": "hours_worked,productivity_score\n5,6\n6,7\n7,8\n8,8\n9,7\n10,6",
    "Sleep + Supplement (T-test)": "day,supplement,hours sleep\nMonday,yes,7.5\nMonday,no,5.5\nTuesday,yes,8.3\nWednesday,no,5.1\nThursday,yes,7.9\nFriday,no,6.5",
    "Meditation & Stress Levels (T-test)": "meditated,stress_level\nyes,3\nno,7\nyes,2\nno,8\nyes,4\nno,6",
    "Caffeine & Focus (T-test)": "caffeine,focus_score\nyes,8\nno,6\nyes,7\nno,5\nyes,9\nno,4",
    "Snack Type & Satisfaction (ANOVA)": "snack,satisfaction_score\nfruit,7\nchips,5\nnuts,6\nfruit,8\nchips,4\nnuts,7",
    "Exercise Type & Mood (Chi-square)": "exercise_type,mood\nyoga,calm\nrunning,energized\nweights,strong\nyoga,calm\nrunning,stressed\nweights,strong",
    "Music Genre & Concentration (Chi-square)": "music_genre,concentration\nclassical,high\npop,medium\nrock,low\nclassical,high\npop,medium\nrock,low"
    }
    sample_name = st.sidebar.selectbox("Choose a sample dataset", list(sample_datasets.keys()))
    data = pd.read_csv(StringIO(sample_datasets[sample_name]))

        
# --- Sidebar for input method ---
st.sidebar.header("Feeling Stuck?")

with st.sidebar.expander("🧪 Try One of These Personal Experiments"):
    st.markdown("""
Here are some fun, simple experiments you can try:

- **☕ Coffee & Focus**  
  *Does caffeine help me focus?*  
  Track: `caffeinated (yes/no)`, `focus score (1–10)`

- **🎵 Music & Concentration**  
  *Does music type affect how well I concentrate?*  
  Track: `music type`, `concentration level`

- **💤 Screens & Sleep**  
  *Does screen time before bed hurt my sleep?*  
  Track: `screen before bed (yes/no)`, `sleep hours`

- **🕒 Hours Worked & Productivity**  
  *Is there an optimal amount of work time for me?*  
  Track: `hours worked`, `productivity score`

- **🥑 Snack Type & Satisfaction**  
  *Which snacks keep me full the longest?*  
  Track: `snack type`, `satiety rating`

- **🏃 Walks & Mood**  
  *Does a walk boost my mood?*  
  Track: `walked (yes/no)`, `mood`

Start small—just 7 days of tracking can give surprising insights!
""")

with st.sidebar.expander("📲 Explore Data You May Already Have from Other Apps"):
    st.markdown("Already using a fitness or health app? You can export your data and explore it here.")

    app_choice = st.selectbox("Select a source app:", [
        "Apple Health", "Google Fit", "Strava", "Garmin Connect", "Fitbit"
    ])
    if app_choice == "Apple Health":
        st.markdown("""
### 🍎 Apple Health (iPhone/Apple Watch)
1. Open the **Health** app on your iPhone.
2. Tap your profile icon → **Export All Health Data**.
3. You'll get a `.zip` file containing an `export.xml`.
4. Tools to convert to CSV:
   - [QS Access](https://apps.apple.com/us/app/qs-access/id920297614)
   - [Health Auto Export](https://apps.apple.com/us/app/health-auto-export-to-csv/id1455780541)
        """)

    elif app_choice == "Google Fit":
        st.markdown("""
### 📱 Google Fit
1. Go to [Google Takeout](https://takeout.google.com/).
2. Select only **Google Fit**.
3. Export your data as a `.zip` archive.
4. Open `Daily Summaries.json` or `Sessions.json`, then convert it to CSV using a JSON converter or script.
        """)

    elif app_choice == "Strava":
        st.markdown("""
### 🚴 Strava

**Option 1: Export All Activities**
1. Log in at [Strava.com](https://www.strava.com/).
2. Go to [Account Export](https://www.strava.com/athlete/delete_your_account).
3. Click **“Request your archive”** and wait for the email.
4. Open the ZIP → use `activities.csv` for upload.

**Option 2: Export a Single Activity**
1. Open an activity page.
2. Click “...” → **Export GPX**, or use `/export_tcx` in the URL.
        """)

    elif app_choice == "Garmin Connect":
        st.markdown("""
### ⌚ Garmin Connect
1. Go to [Garmin Connect](https://connect.garmin.com/).
2. Profile → Account Settings → **Export Your Data**.
3. You'll receive a ZIP file via email.
4. Inside, use `Activities.csv` or export individual activity files via the gear icon.
        """)

    elif app_choice == "Fitbit":
        st.markdown("""
### 💤 Fitbit
1. Go to [Fitbit Data Export](https://www.fitbit.com/settings/data/export).
2. Choose a date range (max 31 days).
3. Download the ZIP with CSVs for:
   - Sleep
   - Heart rate
   - Steps
   - Activities
        """)



st.markdown("""
_Not sure what statistical significance or p-values mean? Scroll to the bottom of the page for a quick intro and helpful links!_
""")

if data is not None:
    st.subheader("Your Data")
    data = st.data_editor(data, num_rows="dynamic", use_container_width=True)


    colnames = list(data.columns)
    if len(colnames) < 2:
        st.error("Please provide at least two columns in your data.")
    else:
        condition_col = st.selectbox("Select the column for your experimental condition", colnames, index=colnames.index("supplement") if "supplement" in colnames else 0)
        outcome_col = st.selectbox("Select the column for your outcome variable", colnames, index=colnames.index("hours sleep") if "hours sleep" in colnames else 1)

        try:
            data[condition_col] = data[condition_col].astype(str).str.strip()
            data[outcome_col] = data[outcome_col].astype(str).str.strip()

            initial_rows = len(data[outcome_col])
            for col in [condition_col, outcome_col]:
                data[col] = data[col].replace(["", "None", "nan"], pd.NA)
            data = data.dropna(subset=[condition_col, outcome_col])
            removed = initial_rows - len(data[outcome_col])
            st.warning(f"{removed} removed rows")

            if removed > 0:
                st.warning(f"{removed} row(s) were removed because they had missing values in your selected columns.")
            
            data_num = data.copy()
            data_num[outcome_col] = pd.to_numeric(data_num[outcome_col], errors='coerce')
            data_num[condition_col] = pd.to_numeric(data_num[condition_col], errors='coerce')
            is_condition_numeric = data_num[condition_col].notna().all()
            is_outcome_numeric = data_num[outcome_col].notna().all()

            result_text = ""
            plot_buffer = BytesIO()

            st.markdown("### 🎛️ Filter Your Data")

            # For condition_col
            if is_condition_numeric:
                data[condition_col] = pd.to_numeric(data[condition_col], errors='coerce')
                min_val, max_val = data[condition_col].min(), data[condition_col].max()
                condition_range = st.slider(f"Filter {condition_col}:", float(min_val), float(max_val), (float(min_val), float(max_val)))
                data = data[data[condition_col].between(condition_range[0], condition_range[1])]
            else:
                categories = sorted(data[condition_col].dropna().unique())
                selected = st.multiselect(f"Select categories for {condition_col}:", categories, default=categories)
                data = data[data[condition_col].isin(selected)]

            # For outcome_col
            if is_outcome_numeric:
                data[outcome_col] = pd.to_numeric(data[outcome_col], errors='coerce')

                min_val, max_val = data[outcome_col].min(), data[outcome_col].max()
                outcome_range = st.slider(f"Filter {outcome_col}:", float(min_val), float(max_val), (float(min_val), float(max_val)))
                data = data[data[outcome_col].between(outcome_range[0], outcome_range[1])]
            else:
                categories = sorted(data[outcome_col].dropna().unique())
                selected = st.multiselect(f"Select categories for {outcome_col}:", categories, default=categories)
                data = data[data[outcome_col].isin(selected)]
                

            if is_condition_numeric and is_outcome_numeric:
                import numpy as np
                from sklearn.linear_model import LinearRegression

                x = data_num[condition_col].values.reshape(-1, 1)
                y = data_num[outcome_col].values

                # Linear model
                linear_model = LinearRegression().fit(x, y)
                linear_preds = linear_model.predict(x)
                linear_r2 = linear_model.score(x, y)

                # Quadratic model
                x_quad = np.hstack([x, x**2])
                quad_model = LinearRegression().fit(x_quad, y)
                quad_preds = quad_model.predict(x_quad)
                quad_r2 = quad_model.score(x_quad, y)

                # Decide which model fits better using R² and penalty for complexity
                adjusted_r2_linear = 1 - (1 - linear_r2) * (len(y) - 1) / (len(y) - 1 - 1)
                adjusted_r2_quad = 1 - (1 - quad_r2) * (len(y) - 1) / (len(y) - 1 - 2)
                better_model = "quadratic" if adjusted_r2_quad > adjusted_r2_linear + 0.1 else "linear"

                st.subheader("📈 Visualization")
                fig, ax = plt.subplots()
                if better_model == "quadratic":
                    sns.scatterplot(x=data_num[condition_col], y=y, ax=ax)
                    sns.lineplot(x=data_num[condition_col], y=quad_preds, color="black", ax=ax)
                else:
                    sns.regplot(x=data_num[condition_col], y=data_num[outcome_col], ax=ax)

                fig.savefig(plot_buffer, format="png")
                st.pyplot(fig)
                
                st.subheader("📖 How to Read This Plot")
                if better_model == "quadratic":
                    st.markdown("This plot uses a **quadratic regression**, which fits a curve to the data. This is useful when relationships are U-shaped or peak-shaped.\n\n- Each point represents one observation.\n- The black line is a best-fit line showing the trend.\n- If the points form a rising pattern, the variables are positively correlated.\n- If they fall together, it's a negative correlation.\n- The tighter the points hug the line, the stronger the relationship.")
                else:
                    st.markdown("This plot uses a **linear regression**, which fits a straight line.\n\n- Each point represents one observation.\n- The black line is a best-fit line showing the trend.\n- If the points form a rising pattern, the variables are positively correlated.\n- If they fall together, it's a negative correlation.\n- The tighter the points hug the line, the stronger the relationship.")
                
                st.subheader("🧠 Correlation Test Result")
                if better_model == "quadratic":
                    result_text = (f"A **quadratic model** better fits the relationship between **{condition_col}** and **{outcome_col}**. "
                        f"This suggests a **nonlinear pattern**—such as a peak or dip—rather than a simple upward or downward trend.\n\n"
                        f"**Adjusted R² = {adjusted_r2_quad:.3f}**, meaning the model explains that proportion of variance after correcting for complexity.\n\n"
                        f"This kind of relationship often shows up when there's an **optimal value** (like productivity peaking at a certain number of hours).")
                    st.info(result_text)
                else:
                    corr, pval = stats.pearsonr(data_num[condition_col], data_num[outcome_col])

                    result_text = f"There is a Pearson correlation of {corr:.3f} between {condition_col} and {outcome_col}, with a p-value of {pval:.4f}. "
                    if pval < 0.05:
                        st.success(result_text + "This is considered statistically significant, meaning it's unlikely to have occurred by chance. 🎉")
                    else:
                        st.info(result_text + "This is not statistically significant. It might just be due to random variation.")
                        
                group_sizes = len(data_num[outcome_col])
                if group_sizes < 20:
                    st.warning("⚠️ Warning: One or more groups have fewer than 20 data points. Even if a result is marked 'statistically significant', it may not be meaningful with very small sample sizes. Collecting more data improves reliability.")

                st.markdown("**⚠️ Note:** Correlation does not imply causation. Just because two things are related doesn’t mean one causes the other. Learn more: [Correlation ≠ Causation](https://www.tylervigen.com/spurious-correlations)")

            elif not is_condition_numeric and is_outcome_numeric:
                data[outcome_col] = pd.to_numeric(data_num[outcome_col], errors='coerce')
                groups = data[condition_col].unique()
                st.subheader("📊 Summary Statistics")
                stats_summary = data[[condition_col, outcome_col]].copy()
                stats_summary[outcome_col] = pd.to_numeric(stats_summary[outcome_col], errors='coerce')
                stats_table = stats_summary.groupby(condition_col)[outcome_col].agg(['count', 'mean', 'median', 'std'])
                st.table(stats_table)
                if len(groups) == 2:
                    
                    group1 = data[data[condition_col] == groups[0]][outcome_col].astype(float).dropna()
                    group2 = data[data[condition_col] == groups[1]][outcome_col].astype(float).dropna()
                    tstat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    st.subheader("📈 Visualization")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=data, x=condition_col, y=outcome_col, ax=ax, showfliers=False)
                    sns.pointplot(data=data, x=condition_col, y=outcome_col, ax=ax, ci=None, markers="D", linestyles = "None", color="black")
                    fig.savefig(plot_buffer, format="png")
                    st.pyplot(fig)
                    
                    st.subheader("📖 How to Read This Plot")
                    st.markdown("This plot compares outcomes across different groups.\n\n- The box shows the middle 50% of values (interquartile range).\n- The line in the box is the **median**.\n- The dots represent **mean values** for each group.\n- Error bars show the **standard deviation**.\n- Wider boxes or long whiskers = more variability.\n\n**Tip:** If boxes overlap a lot, the groups might not be very different.")
                    st.subheader("🧠 T-Test Result")
                    result_text = f"The average {outcome_col} was different for each group of {condition_col}. T-test p-value: {pval:.4f}. "
                    if pval < 0.05:
                        st.success(result_text + "This difference is statistically significant. 🎉")
                    else:
                        st.info(result_text + "This difference is not statistically significant.")
                    if stats_table['count'].min() < 20:
                        st.warning("⚠️ Warning: One or more groups have fewer than 20 data points. Even if a result is marked 'statistically significant', it may not be meaningful with very small sample sizes. Collecting more data improves reliability.")
                else:
                    group_lists = [data[data[condition_col] == g][outcome_col].astype(float).dropna() for g in groups]
                    fstat, pval = f_oneway(*group_lists)
                    st.subheader("📈 Visualization")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=data, x=condition_col, y=outcome_col, ax=ax, showfliers=False)
                    sns.pointplot(data=data, x=condition_col, y=outcome_col, ax=ax, ci=None, markers="D", linestyles = "None", color="black")
                    fig.savefig(plot_buffer, format="png")
                    st.pyplot(fig)
                    st.subheader("📖 How to Read This Plot")
                    st.markdown("This plot compares outcomes across different groups.\n\n- The box shows the middle 50% of values (interquartile range).\n- The line in the box is the **median**.\n- The dots represent **mean values** for each group.\n- Error bars show the **standard deviation**.\n- Wider boxes or long whiskers = more variability.\n\n**Tip:** If boxes overlap a lot, the groups might not be very different.")
                    st.subheader("🧠 ANOVA Test Result")
                    result_text = f"There are multiple groups in {condition_col}. ANOVA F-statistic: {fstat:.2f}, p-value: {pval:.4f}. "
                    if pval < 0.05:
                        st.success(result_text + "At least one group differs significantly. 🎉")
                    else:
                        st.info(result_text + "No significant differences detected between the groups.")
                    if stats_table['count'].min() < 20:
                        st.warning("⚠️ Warning: One or more groups have fewer than 20 data points. Even if a result is marked 'statistically significant', it may not be meaningful with very small sample sizes. Collecting more data improves reliability.")

            elif not is_condition_numeric and not is_outcome_numeric:
                contingency = pd.crosstab(data[condition_col], data[outcome_col])
                chi2, pval, dof, expected = chi2_contingency(contingency)
                st.subheader("📈 Visualization")
                fig, ax = plt.subplots()
                sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", ax=ax)
                fig.savefig(plot_buffer, format="png")
                st.pyplot(fig)
                st.subheader("📖 How to Read This Plot")
                st.markdown("This grid shows how often each combination of categories occurred.\n\n- Darker cells = more observations.\n- Rows = one variable (e.g., treatment type)\n- Columns = the other variable (e.g., outcome)\n- If most of the values cluster in a few squares, the variables might be related.\n- A Chi-square test checks whether the row and column variables are independent or not.\n\n**Look for patterns like: \"this group almost always had this outcome.\"**\"")
                
                st.subheader("🧠 Chi-Square Test Result")
                result_text = f"There is a relationship between {condition_col} and {outcome_col}. Chi-square statistic: {chi2:.2f}, p-value: {pval:.4f}. "
                if pval < 0.05:
                    st.success(result_text + "This relationship is statistically significant. 🎉")
                else:
                    st.info(result_text + "No statistically significant relationship was found.")
                group_counts = contingency.sum(axis=1)  # total per condition group
                if group_counts.min() < 20:
                    st.warning("⚠️ Warning: One or more groups have fewer than 20 data points. Even if a result is marked 'statistically significant', it may not be meaningful with very small sample sizes. Collecting more data improves reliability.")
            else:
                st.error("Unsupported combination. Please double-check your variable types.")

            # --- Downloadable content ---
            st.markdown("### 📥 Export Your Results")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download your dataset as CSV", csv, "experiment_data.csv", "text/csv")

            if plot_buffer:
                st.download_button("Download plot as PNG", plot_buffer.getvalue(), "result_plot.png", "image/png")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    st.markdown("""
---
### 📚 Want to Learn More?
- [What is a p-value? (with examples)](https://www.statsdirect.com/help/basics/p_values.htm)
- [Understanding statistical significance](https://www.scribbr.com/statistics/statistical-significance/)
- [Beginner-friendly intro to t-tests](https://www.scribbr.com/statistics/t-test/)
- [Pearson correlation explained](https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/)
- [Statistics How To – Polynomial Regression](https://www.statisticshowto.com/polynomial-regression/)
- [Chi-square test overview](https://www.scribbr.com/statistics/chi-square-test/)
- [Intro to ANOVA](https://www.scribbr.com/statistics/anova/)
- [Correlation ≠ Causation (with fun examples)](https://www.tylervigen.com/spurious-correlations)

These resources are not required, but if you're curious, they'll help you better understand your results.
""")


