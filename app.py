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
Upload a CSV or manually enter your data below.

_Not sure what statistical significance or p-values mean? Scroll to the bottom of the page for a quick intro and helpful links!_
""")

# --- Sidebar for input method ---
input_method = st.sidebar.radio("How would you like to enter your data?", ["Manual Entry", "Upload CSV"])
sample_data_option = st.sidebar.selectbox("Try an example dataset:", [
    "None",
    "Sleep + Supplement (T-test)",
    "Sleep by Day (ANOVA)",
    "Sleep Quality & Supplement (Chi-square)",
    "Sleep Hours vs Energy (Correlation)",
    "Caffeine & Focus (T-test)",
    "Exercise Type & Mood (Chi-square)",
    "Hours Worked vs Productivity (Correlation)",
    "Snack Type & Satisfaction (ANOVA)",
    "Music Genre & Concentration (Chi-square)",
    "Screen Time vs Happiness (Correlation)",
    "Meditation & Stress Levels (T-test)",
    "Meal Timing & Energy (ANOVA)"
])

# --- Preloaded samples ---
sample_datasets = {
    "Sleep + Supplement (T-test)": "day,supplement,hours sleep\nMonday,yes,7.5\nMonday,no,5.5\nTuesday,yes,8.3\nWednesday,no,5.1\nThursday,yes,7.9\nFriday,no,6.5",
    "Sleep by Day (ANOVA)": "day,hours sleep\nMonday,6.5\nTuesday,7.2\nWednesday,6.8\nThursday,7.0\nFriday,8.1\nSaturday,8.4\nSunday,7.5",
    "Sleep Quality & Supplement (Chi-square)": "supplement,sleep_quality\nyes,good\nyes,excellent\nno,poor\nno,fair\nyes,good\nno,fair",
    "Sleep Hours vs Energy (Correlation)": "energy_level,hours sleep\n5,6.0\n6,6.5\n7,7.2\n8,7.8\n9,8.1\n10,8.4",
    "Caffeine & Focus (T-test)": "caffeine,focus_score\nyes,8\nno,6\nyes,7\nno,5\nyes,9\nno,4",
    "Exercise Type & Mood (Chi-square)": "exercise_type,mood\nyoga,calm\nrunning,energized\nweights,strong\nyoga,calm\nrunning,stressed\nweights,strong",
    "Hours Worked vs Productivity (Correlation)": "hours_worked,productivity_score\n5,6\n6,7\n7,8\n8,8\n9,7\n10,6",
    "Snack Type & Satisfaction (ANOVA)": "snack,satisfaction_score\nfruit,7\nchips,5\nnuts,6\nfruit,8\nchips,4\nnuts,7",
    "Music Genre & Concentration (Chi-square)": "music_genre,concentration\nclassical,high\npop,medium\nrock,low\nclassical,high\npop,medium\nrock,low",
    "Screen Time vs Happiness (Correlation)": "screen_time,happiness_score\n2,8\n4,7\n6,6\n8,5\n10,4\n12,3",
    "Meditation & Stress Levels (T-test)": "meditated,stress_level\nyes,3\nno,7\nyes,2\nno,8\nyes,4\nno,6",
    "Meal Timing & Energy (ANOVA)": "meal_time,energy_score\nmorning,8\nafternoon,7\nevening,5\nmorning,9\nafternoon,6\nevening,4"
}

# --- Data loading ---
data = None
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your file (CSV, TSV, XLS, XLSX)", type=["csv", "tsv", "xls", "xlsx"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        try:
            if file_type == 'csv':
                data = pd.read_csv(uploaded_file)
            elif file_type == 'tsv':
                data = pd.read_csv(uploaded_file, sep='\t')
            elif file_type in ['xls', 'xlsx']:
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

elif sample_data_option != "None":
    from io import StringIO
    data = pd.read_csv(StringIO(sample_datasets[sample_data_option]))
else:
    st.markdown("### Manual Entry")

    manual_data = st.text_area("Enter your data as CSV (with headers)",
                               "eating a banana,hours of sleep\nyes,7.5\nno,6.0\nyes,8.2\nno,5.5")

   # manual_data = st.text_area("Enter your data as CSV (with headers)",
       #                        "day,supplement,hours sleep,sleep quality,energy level\nMonday,yes,7.5,good,8\nMonday,no,5.5,poor,5\nTuesday,yes,8.3,excellent,9\nWednesday,yes,8.5,excellent,9\nWednesday,no,5.1,poor,4\nThursday,yes,7.9,good,7\nThursday,no,6.0,poor,5\nFriday,yes,7.8,good,8\nFriday,no,6.5,fair,6\nSaturday,no,6.4,fair,6\nSaturday,yes,7.2,good,7\nSunday,no,6.1,fair,5\nSunday,yes,7.6,good,8")
    try:
        from io import StringIO
        data = pd.read_csv(StringIO(manual_data))
    except Exception as e:
        st.error("Failed to parse your input. Check format.")

if data is not None:
    st.subheader("Your Data")
    #st.dataframe(data)
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

            data_num = data.copy()
            data_num[outcome_col] = pd.to_numeric(data_num[outcome_col], errors='coerce')
            data_num[condition_col] = pd.to_numeric(data_num[condition_col], errors='coerce')

            is_condition_numeric = data_num[condition_col].notna().all()
            is_outcome_numeric = data_num[outcome_col].notna().all()

            result_text = ""
            plot_buffer = BytesIO()

            if is_condition_numeric and is_outcome_numeric:
                corr, pval = stats.pearsonr(data_num[condition_col], data_num[outcome_col])

                fig, ax = plt.subplots()
                sns.regplot(x=data_num[condition_col], y=data_num[outcome_col], ax=ax)
                fig.savefig(plot_buffer, format="png")
                st.pyplot(fig)
                st.subheader("🧠 Correlation Test Result")
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
                    sns.pointplot(data=data, x=condition_col, y=outcome_col, ax=ax, ci="sd", markers="D", linestyles = "None", color="black")
                    fig.savefig(plot_buffer, format="png")
                    st.pyplot(fig)
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
                    sns.pointplot(data=data, x=condition_col, y=outcome_col, ax=ax, ci="sd", markers="D", linestyles = "None", color="black")
                    fig.savefig(plot_buffer, format="png")
                    st.pyplot(fig)
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
- [Chi-square test overview](https://www.scribbr.com/statistics/chi-square-test/)
- [Intro to ANOVA](https://www.scribbr.com/statistics/anova/)
- [Correlation ≠ Causation (with fun examples)](https://www.tylervigen.com/spurious-correlations)

These resources are not required, but if you're curious, they'll help you better understand your results.
""")


