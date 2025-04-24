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
Upload a file or manually enter your data below.

_Not sure what statistical significance or p-values mean? Scroll to the bottom of the page for a quick intro and helpful links!_
""")

# --- Sidebar for input method ---
input_method = st.sidebar.radio("How would you like to enter your data?", ["Manual Entry", "Upload File"])
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
    "Sleep Hours vs Energy (Correlation)": "energy_level,hours sleep\n5,6.0\n6,6.5\n7,7.2\n8,7.8\n9,8.1\n10,8.4",
    "Hours Worked vs Productivity (Correlation)": "hours_worked,productivity_score\n5,6\n6,7\n7,8\n8,8\n9,7\n10,6",
    "Sleep + Supplement (T-test)": "day,supplement,hours sleep\nMonday,yes,7.5\nMonday,no,5.5\nTuesday,yes,8.3\nWednesday,no,5.1\nThursday,yes,7.9\nFriday,no,6.5",
    "Meditation & Stress Levels (T-test)": "meditated,stress_level\nyes,3\nno,7\nyes,2\nno,8\nyes,4\nno,6",
    "Caffeine & Focus (T-test)": "caffeine,focus_score\nyes,8\nno,6\nyes,7\nno,5\nyes,9\nno,4",
    "Snack Type & Satisfaction (ANOVA)": "snack,satisfaction_score\nfruit,7\nchips,5\nnuts,6\nfruit,8\nchips,4\nnuts,7",
    "Exercise Type & Mood (Chi-square)": "exercise_type,mood\nyoga,calm\nrunning,energized\nweights,strong\nyoga,calm\nrunning,stressed\nweights,strong",
    "Music Genre & Concentration (Chi-square)": "music_genre,concentration\nclassical,high\npop,medium\nrock,low\nclassical,high\npop,medium\nrock,low"
}

# --- Data loading ---
data = None
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "tsv", "xls", "xlsx", "pkl", "parquet"])
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


