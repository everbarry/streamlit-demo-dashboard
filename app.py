import streamlit as st
import numpy as np
import random
from neo4jDriver import Neo4jConnection
import plotly.express as px
from plotly_calplot import calplot, month_calplot
import pandas as pd
from tqdm import tqdm
import datetime


### QUERY DATA STUFF ###

def demoqueries(time: str, lim=None):
    air = {'KLM': 56377143}
    # air = {'KLM': 56377143, 'AirFrance': 106062176, 'British_Airways': 18332190, 'AmericanAir': 22536055,
    #        'Lufthansa': 124476322, 'easyJet': 38676903, 'RyanAir': 1542862735, 'SingaporeAir': 253340062,
    #        'Qantas': 218730857, 'EtihadAirways': 45621423, 'VirginAtlantic': 20626359}
    airSpain = {'KLM': 56377143, 'AirFrance': 106062176, 'easyJet': 38676903, 'RyanAir': 1542862735}
    qs = []
    for airline in air:
        query = 'MATCH p=(a:Tweet)-[:REPLY*1..]->(b:Tweet{author:"' + str(
            air[airline]) + '"})-[:REPLY*0..]->() WHERE a.timestamp =~ "' + time + '.*' + '" RETURN NODES(p)'
        qs.append(query)
    qs = [q + ' LIMIT ' + str(lim) if lim is not None else q for q in qs]
    return qs


@st.experimental_singleton
def demodata(time: str, lim=None):
    dfs = []
    queries = demoqueries(time, lim)
    for query in queries:
        records = conn.query(query)
        df_list = []
        for indx, record in tqdm(enumerate(records), total=len(records)):
            convdf = pd.DataFrame.from_records(record[0])
            convdf["conv"] = indx
            convdf['author'] = convdf['author'].astype(int)
            convdf['score'] = convdf['score'].astype(float)
            convdf['score'] = convdf['score'].astype(int)
            convdf['diff'] = convdf['score'].diff()
            convdf['perc'] = convdf['score'].pct_change(periods=1, fill_method='pad', limit=None, freq=None)
            df_list.append(convdf)
        dfs.append(pd.concat(df_list))
    return dfs


### OLD STUFF THAT WE GOTTA STICK WITH ###

def queryAirlineConvos(time, airline, limit=None, es=False):
    r = []
    query = 'MATCH p=(a:Tweet)-[:REPLY*1..]->(b:Tweet{author:"' + str(
        airline) + '"})-[:REPLY*0..]->() WHERE a.timestamp =~ "' + time + '.*' + '" RETURN p'
    if es:
        query = 'MATCH p=(a:Tweet)-[:REPLY*1..]->(b:Tweet{author:"' + str(
            airline) + '"})-[:REPLY*0..]->() WHERE a.timyearestamp =~ "' + time + '.*' + '" AND a.lang = "es" RETURN p'
    query = query + ' LIMIT ' + str(limit) if limit is not None else query
    res = conn.query(query)
    if res is not None:
        print(f'query executed: {query}')
        for i in res:
            r.append(clean(i[0], str(airline)))
    return r


def clean(d, author='56377143'):
    r = []
    a = d.nodes
    for x in a:
        if x['author'] != author:
            try:
                r.append(int(float(x['score'])))
            except TypeError:
                print('non graded tweet found')
    if not r:
        return [0]
    return r


def makeDiff(l1):
    l2 = l1[1:]
    l1.pop()
    return sum([y - x for x in l1 for y in l2])


def queryAirlines(time, lim=None):
    air = {'KLM': 56377143, 'AirFrance': 106062176, 'British_Airways': 18332190, 'AmericanAir': 22536055,
           'Lufthansa': 124476322, 'easyJet': 38676903, 'RyanAir': 1542862735, 'SingaporeAir': 253340062,
           'Qantas': 218730857, 'EtihadAirways': 45621423, 'VirginAtlantic': 20626359}
    airDiff = {'KLM': [], 'AirFrance': [], 'British_Airways': [], 'AmericanAir': [], 'Lufthansa': [], 'easyJet': [],
               'RyanAir': [], 'SingaporeAir': [], 'Qantas': [], 'EtihadAirways': [], 'VirginAtlantic': []}
    for airline in air:
        air[airline] = queryAirlineConvos(time, air[airline], limit=lim)
    for airline in air:
        for convo in air[airline]:
            airDiff[airline].append(makeDiff(convo))
    for airline in air:
        airDiff[airline] = sum(airDiff[airline]) / len(airDiff[airline])
    return airDiff


### PLOT Functions ###

def plotDist(df):
    colors = ['#A56CC1', '#A6ACEC', '#245551', '#835AF1', '#7FA6EE', '#962fbf']
    color = random.sample(colors, 1)
    df = df[df.author != 56377143]
    df = df.groupby("conv")[["diff", "lang"]].agg({"diff": "mean", "lang": lambda x: x.iloc[0]})
    fig = px.histogram(df, x="diff", marginal="violin", nbins=25, color_discrete_sequence=color,
                       labels={"convdiff": "Sentiment Difference", },
                       title="Distribution of Sentiment Evolution in Conversation")
    return fig


def plotDistEs(df):
    colors = ['#A56CC1', '#A6ACEC', '#245551', '#835AF1', '#7FA6EE', '#962fbf']
    color = random.sample(colors, 1)
    # df = df[df.author != 56377143]
    df = df[df.lang == 'es']
    df = df.groupby("conv")[["diff", "lang"]].agg({"diff": "mean", "lang": lambda x: x.iloc[0]})
    fig = px.histogram(df, x="diff", marginal="violin", nbins=25, color_discrete_sequence=color,
                       labels={"convdiff": "Sentiment Difference", },
                       title="Distribution of Sentiment Evolution in Spanish Conversations")
    return fig


def plotDistPerc(df):
    colors = ['#A56CC1', '#A6ACEC', '#245551', '#835AF1', '#7FA6EE', '#962fbf']
    color = random.sample(colors, 1)
    df = df[df.author != 56377143]
    df = df.groupby("conv")[["perc", "lang"]].agg({"perc": "mean", "lang": lambda x: x.iloc[0]})
    fig = px.histogram(df, x="perc", marginal="violin", nbins=25, color_discrete_sequence=color,
                       labels={"convdiffperc": "Percentage Sentiment Difference", },
                       title="Distribution of Percentage Sentiment change throughout in Conversation")
    return fig


def barPlot(dct):
    colors = ['#A56CC1', '#A6ACEC', '#245551', '#835AF1', '#7FA6EE', '#962fbf']
    color = random.sample(colors, 1)
    dct = dict(sorted(dct.items(), key=lambda x: x[1]))
    fig = px.bar(y=[dct[i] for i in dct], x=dct.keys(), color_discrete_sequence=color,
                 labels={"convdiff": "Mean Sentiment Difference"},
                 title="Mean Sentiment Difference Evolution in Conversations for airlines")
    return fig


def stackedBar(df):
    df = df[df.author != 56377143]
    df = df.groupby("conv")[["diff", "lang"]].agg({"diff": "mean", "lang": lambda x: x.iloc[0]})
    dfconv = pd.concat([df])
    intlang = ["en", "es", "nl", "de", "fr", "it"]
    fig = px.histogram(dfconv[dfconv["lang"].isin(intlang)], x="diff", marginal="box", color="lang", nbins=25,
                       labels={"convdiff": "Sentiment Difference", },
                       title="KLM Langauge performance (Showcase of Multilingual model)", height=800)
    return fig


def season(df):
    df = df[df.author != 56377143]
    dfklm = df[["conv", "timestamp", "diff"]].agg(
        {"conv": lambda x: x.iloc[0], "timestamp": lambda x: x.iloc[0], "diff": "mean"})
    dfklm['date'] = pd.to_datetime(dfklm["timestamp"])
    dfklm['date'] = pd.to_datetime(dfklm['date'])
    dfklm['diff'] = dfklm['diff'] + 4.0
    fig = calplot(dfklm, x="date", y="diff", colorscale="thermal", gap=0, month_lines_width=3, month_lines_color="#fff",
                  space_between_plots=0.25, years_title=True)
    return fig


### STREAMLIT Config ###

conn = Neo4jConnection(uri="bolt://139.162.187.122:7687", user="neo4j", pwd="test123")
st.title('Demo')
limit = None
date = '2020-01'
d = st.date_input(
     "Click on any day to select the month",
     datetime.date(2020, 1, 1))
date = str(d)[:7]
st.write('Your chosen date is:', date)
data_load_state = st.text('Loading data...')
dfs = demodata(date, lim=limit)
klm = dfs[0]
data_load_state.text('Loading data...done!')
st.plotly_chart(plotDist(klm), use_container_width=True)
st.plotly_chart(plotDistPerc(klm), use_container_width=True)
st.plotly_chart(stackedBar(klm), use_container_width=True)
st.plotly_chart(plotDistEs(klm), use_container_width=True)
st.plotly_chart(barPlot(queryAirlines(date, lim=limit)), use_container_width=True)
# st.plotly_chart(season(klm))
