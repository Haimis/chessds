from matplotlib.figure import Figure
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date
from datetime import datetime
from matplotlib.figure import Figure
import base64
from io import BytesIO
import requests
import chess.pgn
import pandas as pd
import os

def build_df(export_url):
    records = []
    print('download starts')
    res = requests.get(export_url, allow_redirects=True)
    print('download status code: ', res.status_code)
    open('data/file.pgn', 'wb').write(res.content)
    with open('data/file.pgn', 'r') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            record = {}
            for key in game.headers:
                record[key] = game.headers[key]
            records.append(record)
    df = pd.DataFrame.from_records(records)
    os.remove('data/file.pgn')
    return df

def build_data(df, player):
    df = df.dropna()
    df = df[df['BlackElo'] != '?']
    df = df[df['WhiteElo'] != '?']
    def classify_results(white, result):
        if white == player:
            if result == '1-0':
                return 1
            elif result == '0-1':
                return -1
            else:
                return 0

        if result == '1-0':
            return -1
        elif result == '0-1':
            return 1
        else:
            return 0

    df['UTCDate'] = df.apply(lambda x: f"{x['UTCDate']} {x['UTCTime']}", axis=1) 


    df = df.astype({
    'WhiteRatingDiff': 'int64',
    'BlackRatingDiff': 'int64',
    'Date': 'datetime64',
    'UTCDate': 'datetime64',
    'UTCTime': 'datetime64',
    'WhiteElo': 'int64',
    'BlackElo': 'int64',
    })

#    print(df['WhiteRatingDiff'].unique())
#    print(df['BlackRatingDiff'].unique())

    openings = list(df['Opening'].unique())
    df['OpeningCode'] = df['Opening'].map(lambda x: openings.index(x))
    df['Hour'] = df['UTCTime'].map(lambda x: int(str(x).split(':')[0][-2:]))
    df['Advance'] = df.apply(
        lambda x: x['WhiteElo'] - x['BlackElo'] if x['White'] == player 
        else x['BlackElo'] - x['WhiteElo'], axis=1)
    df['Victory'] = df.apply(lambda x: classify_results(x['White'], x['Result']), axis=1)

    #df = df[df['Y'] != 0]
    df['MyDiff'] = df.apply(
        lambda x: x['WhiteRatingDiff'] if x['White'] == player else x['BlackRatingDiff'], axis=1)
    df['MyElo'] = df.apply(
        lambda x: x['WhiteElo'] if x['White'] == player else x['BlackElo'], axis=1)
    df['OppElo'] = df.apply(
        lambda x: x['WhiteElo'] if x['White'] != player else x['BlackElo'], axis=1)
    df['Color'] = df['White'].map(lambda x: 0 if x == player else 1)
#    print(df.iloc[0])
    data = df[[
        'Color',
        'MyElo',
        'OppElo',
        'Hour',
        'Advance',
        'OpeningCode',
        'Victory'
    ]]

    df = df.sort_values(by='UTCDate')
    df = df.set_index('UTCDate')
    return df

def get_basic_data(df):

    df_white = df[df['Color'] == 0]
    df_black = df[df['Color'] == 1]
    white_win = len(df_white[df_white['Victory'] == 1])
    white_tie = len(df_white[df_white['Victory'] == 0])
    white_los = len(df_white[df_white['Victory'] == -1])
    black_win = len(df_black[df_black['Victory'] == 1])
    black_tie = len(df_black[df_black['Victory'] == 0])
    black_los = len(df_black[df_black['Victory'] == -1])
    w_per_win = "%.2f" % (white_win/len(df_white)*100)
    w_per_tie = "%.2f" % (white_tie/len(df_white)*100)
    w_per_los = "%.2f" % (white_los/len(df_white)*100)
    b_per_win = "%.2f" % (black_win/len(df_black)*100)
    b_per_tie = "%.2f" % (black_tie/len(df_black)*100)
    b_per_los = "%.2f" % (black_los/len(df_black)*100)
    per_win = "%.2f" % (len(df[df["Victory"] == 1])/len(df)*100)
    per_tie = "%.2f" % (len(df[df["Victory"] == 0])/len(df)*100)
    per_los = "%.2f" % (len(df[df["Victory"] == -1])/len(df)*100)

    headings = ['Color', 'Won', 'Tie', 'Loss', 'Games played']

    data = [
        [
            'White', 
            f'{white_win} ({w_per_win} %)',
            f'{white_tie} ({w_per_tie} %)',
            f'{white_los} ({w_per_los} %)',
            len(df_white)
        ],
        [
            'Black', 
            f'{black_win} ({b_per_win} %)',
            f'{black_tie} ({b_per_tie} %)',
            f'{black_los} ({b_per_los} %)',
            len(df_black)
        ],
        [
            'Total', 
            f'{len(df[df["Victory"] == 1])} ({per_win} %)',
            f'{len(df[df["Victory"] == 0])} ({per_tie} %)',
            f'{len(df[df["Victory"] == -1])} ({per_los} %)',
            len(df)
        ]
    ]
    
    return headings, data

def game_by_hour(df):
    y1 = df.groupby('Hour', as_index=False)['Hour'].count()
    y2 = df.groupby('Hour', as_index=False)['MyDiff'].mean()
    fig = Figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    ax1.plot(y1.index, y1, 'g-')
    ax2.plot(y2['Hour'], y2['MyDiff'], 'b-')
    ax2.hlines(df['MyDiff'].mean(), xmin=0, xmax=23, label = 'average pts per game', colors='Red')
    #
    ax1.set_xlabel('Hour (UTC)')
    ax1.set_ylabel('Games played this hour', color='g')
    ax2.set_ylabel('Mean elo gain this hour', color='b')
    fig.legend(loc="lower right")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getbuffer()).decode("ascii")
    

def monthly_mean(df):
    fig = Figure()
    ax1 = fig.subplots()
    df = df['MyElo'].groupby([df.index.year, df.index.month]).aggregate(['count', 'mean'])
    df['mean'].plot(ax=ax1, color='black')
    ax2 = ax1.twinx()
    df['count'].plot(ax=ax2, color='lightgrey')
    ax1.set_ylabel('monthly mean elo')
    ax1.set_xlabel('time')
    ax2.set_ylabel('games per month')
    ax1.legend([ax1.get_lines()[0], ax2.get_lines()[0]], ['mean','games'])
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getbuffer()).decode("ascii")

def get_openings(df):
    ops = df.groupby('Opening')['MyDiff'].aggregate(['count', 'mean'])
    ops = ops[ops['count'] >= len(df)/50]
    ops = ops.sort_values(by='mean', ascending=False)
    best_and_worst = ops[['mean', 'count']].head().append(ops[['mean', 'count']].tail())
    best_and_worst['mean'] = best_and_worst['mean'].map(lambda x: "%.2f" % x)
    best_and_worst['opening'] = list(best_and_worst.index)
    best_and_worst['link'] = best_and_worst['opening'].map(lambda x: f'https://en.wikipedia.org/wiki/' + x.split(':')[0].replace(' ', '_'))
    best_and_worst.index = best_and_worst['opening'] + ': ' + best_and_worst['count'].astype(str) + ' game(s)'
    best_and_worst['average'] = best_and_worst['mean']
    best_and_worst = best_and_worst[[
        'link',
        'count',
        'average',
        'opening'
    ]]

    data = best_and_worst.to_records()
    best_and_worst['average'] = best_and_worst['average'].astype(float)
    best_and_worst = best_and_worst.sort_values(by='average', ascending=True)
    headers = ['Opening name', 'Games played', 'Average elo gain per game']
    

    fig = Figure()
    ax = fig.subplots()
    ax.barh(best_and_worst['opening'], best_and_worst['average'], align='edge')
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")

    return headers, data, base64.b64encode(buf.getbuffer()).decode("ascii")

def rematch(df, player):
    last_opp = None
    last_res = None
    i = 0

    lost_first = []
    won_first = []
    draw_first = []

    for idx, row in df.iterrows():
        current_opp = row['White']
        if current_opp == player:
            current_opp = row['Black']
        if last_opp == current_opp:
            if last_res == -1:
                lost_first.append((row['Advance'], row['MyDiff']))
            elif last_res == 1:
                won_first.append((row['Advance'], row['MyDiff']))
            else:
                draw_first.append((row['Advance'], row['MyDiff']))
        last_opp = current_opp
        last_res = row['Victory']
        i += 1


    data = {}
    if len(won_first) > 0:
        wd = pd.DataFrame(won_first)[0].mean()
        wpg = pd.DataFrame(won_first)[1].mean()
        wl = len(won_first)
        data['won'] = {
            'd': "%.2f" % wd,
            'pg': "%.2f" % wpg,
            'l': wl
        }
    if len(draw_first) > 0:
        td = pd.DataFrame(draw_first)[0].mean()
        tpg = pd.DataFrame(draw_first)[1].mean()
        tl = len(draw_first)
        data['tied'] = {
            'd': "%.2f" % td,
            'pg': "%.2f" % tpg,
            'l': tl
        }
    if len(lost_first) > 0:
        ld = pd.DataFrame(lost_first)[0].mean()
        lpg = pd.DataFrame(lost_first)[1].mean()
        ll = len(lost_first)
        data['lost'] = {
            'd': "%.2f" % ld,
            'pg': "%.2f" % lpg,
            'l': ll
        }

#    print(data)
    return data

def plot_lr(df):
    fig = Figure()
    ax = fig.subplots()

    y = df['MyElo'].values.reshape(-1, 1)
    X = np.array(df.index.map(lambda x: date.toordinal(x))).reshape(-1, 1)
    reg = LinearRegression().fit(X,y)
    reg.score(X,y)
    Y_pred = reg.predict(X)

    latest_elo = y[len(y) - 1]
    elo_pred = max(0, reg.predict(np.array(X.max()+365).reshape(-1, 1)))
    col = 'red'
    if elo_pred >= latest_elo:
        col = 'green'
    ax.plot(df.index, list(df['MyElo']))
    ax.plot(df.index, Y_pred, color='grey')
    ax.plot([datetime.fromordinal(X.max()), datetime.fromordinal(X.max()+365)], [latest_elo, elo_pred], color=col)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getbuffer()).decode("ascii")
