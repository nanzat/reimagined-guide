import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime


st.title('Quotes Apple')
tickerSymbol = 'AAPL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

st.markdown("""
#### Choose period from 2000-01-01 to the present
""")
period = st.date_input(label="",
                  value=(datetime.datetime.today()-datetime.timedelta(days=365), datetime.datetime.today()),
                    min_value=datetime.date(2000, 1, 1),
                  max_value=datetime.datetime.today(), label_visibility='collapsed')
if len(period)==2:
    if period[1]-period[0]>datetime.timedelta(days=1):
        #get the historical prices for this ticker
        tickerDf = tickerData.history(period='1d', start=period[0], end=period[1])
        # Open	High	Low	Close	Volume	Dividends	Stock Splits

        st.markdown("""
        #### Choose the columns to display
        """)
        options = st.multiselect(
            "Choose the columns to display",
            tickerDf.columns,
            [],
            label_visibility="collapsed"
        )
        for i in options:
            st.write(f"""
            ## {i}
            """)
            st.line_chart(tickerDf[i])
    else:
        st.write(f"""
                    ## Period incorrect
                    """)
