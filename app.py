import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

st.set_page_config(page_title="Run Your Own Experiment", layout="centered")
st.title("🧪 Personal Experiment Analyzer")

st.markdown("""
Use this tool to analyze whether a certain action (like eating a banana before bed) affects an outcome (like sleep hours).
Upload a CSV or manually enter your data below.

_Not sure what statistical significance or p-values mean? Scroll to the bottom of the page for a quick intro and helpful links!_
""")

# --- Sidebar for input method ---
input_method = st.sidebar.radio("How would you like to enter your data?", ["Upload CSV", "Manual Entry"])

# --- Data loading ---
data = None
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
else:
    st.markdown("### Manual Entry")
#    manual_data = st.text_area("Enter your data as CSV (with headers)",
    #                           "condition,outcome\nyes,7.5\nno,6.0\nyes,8.2\nno,5.5")
    manual_data = st.text_area("Enter your data as CSV (with headers)",
                "supplement,day,sleep quality,energy level,hours sleep\n                 yes,Monday,good,8,7.5\n                 no,Monday,poor,5,5.5\n                 yes,Tuesday,excellent,9,8.3\n                yes,Wednesday,excellent,9,8.5\n               no,Wednesday,poor,4,5.1\n                yes,Thursday,good,7,7.9\n                no,Thursday,poor,5,6.0\n                yes,Friday,good,8,7.8\n                no,Friday,fair,6,6.5\n                no,Saturday,fair,6,6.4\n                yes,Saturday,good,7,7.2\n                no,Sunday,fair,5,6.1\n                yes,Sunday,good,8,7.6")

    try:
        from io import StringIO
        data = pd.read_csv(StringIO(manual_data))
    except Exception as e:
        st.error("Failed to parse your input. Check format.")

if data is not None:
    st.subheader("Your Data")
    st.dataframe(data)

    colnames = list(data.columns)
    if len(colnames) < 2:
        st.error("Please provide at least two columns in your data.")
    else:
        condition_col = st.selectbox("Select the column for your experimental condition", colnames)
        outcome_col = st.selectbox("Select the column for your outcome variable", colnames, index=1 if len(colnames) > 1 else 0)

        try:
            data[condition_col] = data[condition_col].astype(str).str.strip()
            data[outcome_col] = data[outcome_col].astype(str).str.strip()

            # Try numeric coercion to check for quantitative values
            data_num = data.copy()
            data_num[outcome_col] = pd.to_numeric(data_num[outcome_col], errors='coerce')
            data_num[condition_col] = pd.to_numeric(data_num[condition_col], errors='coerce')

            is_condition_numeric = data_num[condition_col].notna().all()
            is_outcome_numeric = data_num[outcome_col].notna().all()

            if is_condition_numeric and is_outcome_numeric:
                corr, pval = stats.pearsonr(data_num[condition_col], data_num[outcome_col])
                st.subheader("📈 Visualization")
                fig, ax = plt.subplots()
                sns.regplot(x=data_num[condition_col], y=data_num[outcome_col], ax=ax)
                st.pyplot(fig)
                st.subheader("🧠 Correlation Test Result")
                st.write(f"Pearson correlation: **{corr:.3f}**, p-value: **{pval:.4f}**")
                st.markdown("A **correlation** shows how closely two variables are related. A p-value < 0.05 suggests this relationship likely isn't due to chance.")
                if pval < 0.05:
                    st.success("✅ There is a statistically significant correlation! 🎉")
                else:
                    st.info("⚠️ No significant correlation (p >= 0.05). Keep exploring!")

            elif not is_condition_numeric and is_outcome_numeric:
                groups = data[condition_col].unique()
                if len(groups) == 2:
                    group1 = data[data[condition_col] == groups[0]][outcome_col].astype(float).dropna()
                    group2 = data[data[condition_col] == groups[1]][outcome_col].astype(float).dropna()
                    tstat, pval = stats.ttest_ind(group1, group2, equal_var=False)
                    st.subheader("📈 Visualization")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=data, x=condition_col, y=outcome_col, ax=ax, showfliers=False)
                    sns.pointplot(data=data, x=condition_col, y=outcome_col, ax=ax, ci="sd", markers="D", color="black")
                    st.pyplot(fig)
                    st.subheader("🧠 Statistical Test Result")
                    st.write(f"T-test p-value: **{pval:.4f}**")
                    st.markdown("A **t-test** compares the means of two groups. A low p-value means there's likely a real difference between them.")
                    if pval < 0.05:
                        st.success("✅ Your result is statistically significant (p < 0.05). 🎉")
                    else:
                        st.info("⚠️ Your result is *not* statistically significant (p >= 0.05). Keep collecting data!")
                else:
                    group_lists = [data[data[condition_col] == g][outcome_col].astype(float).dropna() for g in groups]
                    fstat, pval = f_oneway(*group_lists)
                    st.subheader("📈 Visualization")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=data, x=condition_col, y=outcome_col, ax=ax, showfliers=False)
                    sns.pointplot(data=data, x=condition_col, y=outcome_col, ax=ax, ci="sd", markers="D", color="black")
                    st.pyplot(fig)
                    st.subheader("🧠 ANOVA Test Result")
                    st.write(f"ANOVA F-statistic: **{fstat:.2f}**, p-value: **{pval:.4f}**")
                    st.markdown("An **ANOVA** compares the means across more than two groups. A low p-value suggests that at least one group is significantly different.")
                    if pval < 0.05:
                        st.success("✅ At least one group differs significantly (p < 0.05). 🎉")
                    else:
                        st.info("⚠️ No significant group differences found (p >= 0.05).")

            elif not is_condition_numeric and not is_outcome_numeric:
                contingency = pd.crosstab(data[condition_col], data[outcome_col])
                chi2, pval, dof, expected = chi2_contingency(contingency)
                st.subheader("📈 Visualization")
                fig, ax = plt.subplots()
                sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                st.subheader("🧠 Chi-Square Test Result")
                st.write(f"Chi-square: **{chi2:.2f}**, df: {dof}, p-value: **{pval:.4f}**")
                st.markdown("A **Chi-square test** checks if two categories are related. A low p-value suggests there's likely a meaningful connection.")
                if pval < 0.05:
                    st.success("✅ There is a significant relationship between the variables! 🎉")
                else:
                    st.info("⚠️ No significant relationship found (p >= 0.05).")

            else:
                st.error("Unsupported combination. Please double-check your variable types.")

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

These resources are not required, but if you're curious, they'll help you better understand your results.
""")
