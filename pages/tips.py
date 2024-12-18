import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
@st.cache_data
def download_file(p):
    return pd.read_csv(p)

path = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
tips = pd.read_csv(path)
def random_dates(start = pd.to_datetime('2023-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S'),
                 end = pd.to_datetime('2023-01-31 23:59:59', format='%Y-%m-%d %H:%M:%S'), n=244):

    start_u = start.value//10**9
    end_u = end.value//10**9

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

pd.to_datetime('2023-01-01 12:00:00', format='%Y-%m-%d %H:%M:%S')
tips['time_order'] = random_dates()

st.title('Tips')

st.markdown("""
#### Choose a CSV file otherwise the default file will be used
#### Chosen dataframe
""")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",label_visibility='collapsed',type='csv'
)
if uploaded_file is not None:
    upd_tips = download_file(uploaded_file)
    if upd_tips.columns == tips.columns:
        tips = upd_tips


st.dataframe(tips.head())

st.markdown("""
#### In the sidebar you can select the charts to be displayed
""")
st.sidebar.write("""
## Available charts
""")
if st.sidebar.checkbox("Tips by time", key="tips"):
    fig, ax= plt.subplots()
    sns.set_theme(style="darkgrid")
    sns.lineplot(x="time_order", y="tip",
                 data=tips, ax=ax)
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download linechart",
            data=file,
            file_name="Tips_by_time.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Total bill histogram", key="hist_total"):
    fig, ax= plt.subplots()
    sns.set_theme(style="ticks")
    sns.despine(fig)

    sns.histplot(
        tips,
        x="total_bill",
        linewidth=.5,
    )
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download histogram",
            data=file,
            file_name="Total_bill_hist.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Scatterplot total bill and tip", key="Scatterplot total_bill and tip"):


    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x="total_bill", y="tip",
                    hue="day",
                    sizes=(1, 8), linewidth=0,
                    data=tips, ax=ax)
    plt.grid(True)
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download scatterplot",
            data=file,
            file_name="Scatter_total_bill_n_tip.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Scatterplot total bill, tip and size", key="Scatterplot size"):


    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x="total_bill", y="tip",
                    hue="size",
                    sizes=(1, 8), linewidth=0,
                    data=tips, ax=ax)
    plt.grid(True)
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download scatterplot",
            data=file,
            file_name="Scatter_size.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Bar chart total bill by week", key="barchart total bill"):
    df1 = tips.groupby('day')['total_bill'].mean()
    df1 = df1.reset_index()

    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid", rc={'figure.figsize': (11.7, 8.27)})
    sns.barplot(x=df1['day'], y=df1['total_bill'], hue=df1['day'], palette="rocket", width=0.5,ax=ax)
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download bar chart",
            data=file,
            file_name="bar_chart_total.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Scatterplot tips by week", key="tips by week"):
    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x="tip", y="day",
                    hue="sex",
                    sizes=(1, 8), linewidth=0,
                    data=tips, ax=ax)

    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download scatterplot",
            data=file,
            file_name="scatter_tips_by_week.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Box plot sum of bills by week", key="box plot"):
    fig, ax = plt.subplots()
    sns.set_theme(style="ticks", palette="pastel")

    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(x="day", y="total_bill",
                hue="time", palette=["m", "g"],
                data=tips, ax=ax)
    sns.despine(offset=10, trim=True)

    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download box plot",
            data=file,
            file_name="box_plot.png",
            mime="image/png",
        )


if st.sidebar.checkbox("Categorical plot sum of bills by week", key="cat plot"):
    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=tips, kind="bar",
        x="day", y="total_bill", hue="time",
        errorbar="sd", palette="dark", alpha=.6, height=6,
    )

    st.pyplot(g)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download catplot",
            data=file,
            file_name="cat_plot.png",
            mime="image/png",
        )


if st.sidebar.checkbox("Histogram of tips by dinner and lunch", key="Histogram of tips by dinner and lunch"):
    fig, ax = plt.subplots()
    sns.set_theme(style="darkgrid")
    f = sns.displot(
        tips, x="tip", kind='hist', col='time', facet_kws=dict(margin_titles=True),
    )

    st.pyplot(f)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download histogram",
            data=file,
            file_name="hist_tips.png",
            mime="image/png",
        )

if st.sidebar.checkbox("Scatterplot total bill and tips by male and female", key="Scatterplot total bill and tips by male and female"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x="total_bill", y="tip",
                    hue="smoker",
                    sizes=(1, 8), linewidth=0,
                    data=tips.loc[tips['sex'] == 'Male'], ax=ax1)
    sns.scatterplot(x="total_bill", y="tip",
                    hue="smoker",
                    sizes=(1, 8), linewidth=0,
                    data=tips.loc[tips['sex'] == 'Female'], ax=ax2)
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download scatterplot",
            data=file,
            file_name="scatter_male_female.png",
            mime="image/png",
        )


if st.sidebar.checkbox("Heatmap of numeric features", key="Heatmap of numeric features"):
    corr = tips[['total_bill', 'tip', 'size', 'time_order']].corr()
    sns.set_theme(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download heatmap",
            data=file,
            file_name="heatmap.png",
            mime="image/png",
        )


if st.sidebar.checkbox("Comparison of male female tips", key="Comparison of male female tips"):
    df1 = tips.groupby('sex')[['total_bill', 'tip']].mean()
    df1 = df1.reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set_theme(style="darkgrid", rc={'figure.figsize': (11.7, 8.27)})
    sns.barplot(x=df1['sex'], y=df1['total_bill'], hue=df1['sex'], palette="rocket", ax=ax1, width=0.5)
    sns.barplot(x=df1['sex'], y=df1['tip'], hue=df1['sex'], palette="rocket", ax=ax2, width=0.5)
    st.pyplot(fig)
    fig.savefig("fig.png")
    with open("fig.png", "rb") as file:
        btn = st.download_button(
            label="Download bar chart",
            data=file,
            file_name="end.png",
            mime="image/png",
        )