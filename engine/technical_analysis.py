import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import mplfinance as mpf
from datetime import datetime, timedelta
# import torch
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('/home/giang/aws_hackathon/logs/technical_analysis.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

root = "/home/giang/aws_hackathon/data"
signal_file = "/home/giang/aws_hackathon/data/signal.txt"
#
def plotCandlestick(df, _predict, alpha=0.3):
    """
    df schema: Timestamp(index), Open, High, Low, Close, Volume
    :param df:
    :param _predict:
    :param alpha:
    :return:
    """
    fig, axs = plt.subplots(2, 1)
    ax = axs[0]
    # print(predict)
    predict = pd.DataFrame(_predict, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    # last timestamp df in index
    last_timestamp = df.index[-1]
    # add timestamp to predict_df by increase seq with step = +4h
    predict['Timestamp'] = pd.date_range(start=last_timestamp + timedelta(hours=4), periods=len(predict), freq='4H')
    # set timestamp to index
    predict.set_index('Timestamp', inplace=True)
    # add the first row to predict_df by index last_timestamp with value of df last row in df
    predict = pd.concat([df.tail(1), predict])

    # define width of candlestick elements
    width = .15
    width2 = .02

    col = 'orange'
    # add second line to plot
    all = df[df.Volume > 0]
    ax.bar(all.index, all.Volume, color=col)

    # add second y-axis label
    ax.set_xlabel('Timestamp', fontsize=14)
    ax.set_ylabel('Volume', color=col, fontsize=14)

    ax2 = ax.twinx()
    ax2.set_ylabel('Price', fontsize=14)

    # define colors to use
    col1 = 'green'
    col2 = 'red'
    up = df[df.Close >= df.Open]
    down = df[df.Close < df.Open]

    ax2.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color=col1)
    ax2.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color=col1)
    ax2.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color=col1)

    # plot down prices
    ax2.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color=col2)
    ax2.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color=col2)
    ax2.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color=col2)

    axs[1].plot(df.index, df.Close, color='black')
    axs[1].plot(predict.index, predict.Close, color='red', linestyle='dashed')
    axs[1].set_xlabel('Timestamp', fontsize=14)
    axs[1].set_ylabel('Price', fontsize=14)
    axs[1].set_title('Close price ', fontsize=16)

    up = predict[predict.Close >= predict.Open]
    down = predict[predict.Close < predict.Open]

    # plot up prices
    ax2.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color=col1, alpha=alpha)
    ax2.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color=col1, alpha=alpha)
    ax2.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color=col1, alpha=alpha)

    # plot down prices
    ax2.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color=col2, alpha=alpha)
    ax2.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color=col2, alpha=alpha)
    ax2.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color=col2, alpha=alpha)

    return fig, ax, ax2

# def plotCandlestick(df, _predict, alpha=0.3):
#     """
#     df schema: Timestamp(index), Open, High, Low, Close, Volume
#     :param df: Historical data DataFrame
#     :param _predict: Predicted data (same schema as df)
#     :param alpha: Transparency of predicted bars
#     :return: Plotly Figure object
#     """
#
#     # Create a copy of the original DataFrame to avoid modifying the original data
#     df = df.copy()
#     _predict = _predict.copy()
#
#     # Add timestamp to predict DataFrame by increasing seq with step = +4h
#     # _predict['Timestamp'] = pd.date_range(start=last_timestamp + timedelta(hours=4), periods=len(_predict), freq='4H')
#     # _predict.set_index('Timestamp', inplace=True)
#
#     _predict = pd.DataFrame(_predict, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
#     # last timestamp df in index
#     last_timestamp = df.index[-1]
#     _predict['Timestamp'] = pd.date_range(start=last_timestamp + timedelta(hours=4), periods=len(_predict), freq='4H')
#
#     # Concatenate the last row of df to _predict to have a continuous plot
#     _tail_df = df.tail(1)
#     # convert index to "Timestamp" column
#     _tail_df.reset_index(inplace=True)
#     # rename column "index" to "Timestamp"
#     _tail_df.rename(columns={'index': 'Timestamp'}, inplace=True)
#
#     # _tail_df.drop(columns=['Timestamp'], inplace=True)
#     predict = pd.concat([_tail_df, _predict])
#
#     # Create the figure
#     # fig = go.Figure()
#     print(predict.head())
#     print(df.head())
#     fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.5, 0.5])
#
#     fig.add_trace(go.Candlestick(x=df.index,
#                                  open=df['Open'],
#                                  high=df['High'],
#                                  low=df['Low'],
#                                  close=df['Close'],
#                                  increasing_line_color='green',
#                                  decreasing_line_color='red',
#                                  showlegend=False,
#                                  name='Candlestick'), row=1, col=1)
#     fig.add_trace(go.Candlestick(x=predict['Timestamp'],
#                                  open=predict['Open'],
#                                  high=predict['High'],
#                                  low=predict['Low'],
#                                  close=predict['Close'],
#                                  increasing_line_color='green',
#                                  decreasing_line_color='red',
#                                  showlegend=False,
#                                  opacity=alpha,
#                                  name='Predicted'), row=1, col=1)
#     fig.update_layout(xaxis_rangeslider_visible=False)
#     fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='orange', name='Volume', opacity=0.7), row=2, col=1)
#     fig.update_yaxes(title_text='Volume', row=2, col=1, secondary_y=True)
#     fig.update_layout(xaxis_title='Timestamp', yaxis_title='Price', title='Candlestick Chart')
#     return fig


class TechnicalAnalysis:
    def __init__(self, json_file):
        self.df = self.getdata(json_file)
        self.df = self.df.iloc[-200:]
        logger.info(self.df.tail(2))
        # print(self.df.tail(2))
        self.predict_price_ = self.get_predict_price()
        self.predict_price = self.predict_price_[0][3]
        # print(self.predict_price)

    def get_predict_price(self):
        # print(self.df.to_numpy().shape)
        result = requests.post("http://10.1.38.211:8888/api/v1/predict",
                               json={'data': self.df.to_numpy().tolist(),
                                     'num_predict': 5})
        if result.status_code == 200:
            print([max(p) for p in zip(*result.json()['prediction'])])
            return result.json()['prediction']
        else:
            logger.error("Error when get predict price")
            return None

    def getdata(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        process_data = [d for d in data if len(d) != 0]

        new_process_data = []
        for d in process_data:
            timestamp = d[0][0]
            open_price = float(d[0][1])
            high = float(d[0][2])
            low = float(d[0][3])
            close = float(d[0][4])
            volume = float(d[0][5])
            new_process_data.append([timestamp, open_price, high, low, close, volume])

        df = pd.DataFrame(new_process_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df.set_index('Timestamp')
        df = df.sort_index()
        logger.info(df.head(30))
        logger.info(df.tail(30))
        return df

    # def update_df(self, json_file):
    #     self.df = self.getdata(json_file)

    def get_df(self):
        return self.df

    def buy_stategy(self, current_price, predict_price, ema, bb_upper, bb_lower, rsi, macd, adx):
        if predict_price is None:
            price_condition = True
        else:
            price_condition = current_price < predict_price
        ema_condition = current_price > ema
        bb_condition = current_price > bb_lower
        rsi_condition = rsi[-1] > 30
        macd_condition = macd[-1] > 0
        adx_condition = adx > 20
        if price_condition and ema_condition and bb_condition and rsi_condition and macd_condition and adx_condition:
            return True
        return False

    def sell_stategy(self, current_price, predict_price, ema, bb_upper, bb_lower, rsi, macd, adx):
        if predict_price is None:
            price_condition = True
        else:
            price_condition = current_price > predict_price
        ema_condition = current_price < ema
        bb_condition = current_price < bb_upper
        rsi_condition = rsi[-1] > 70
        macd_condition = macd[-1] < 0
        adx_condition = adx > 20
        if price_condition and ema_condition and bb_condition and rsi_condition and macd_condition and adx_condition:
            return True
        return False

    def get_buy_sell_signal(self):
        current_price = self.df['Close'].tail(1).values[0]
        ema = self.get_ema_signal(50).tail(1).values[0]
        bb_upper, bb_lower = self.get_bb_signal(50)
        bb_upper = bb_upper.tail(1).values[0]
        bb_lower = bb_lower.tail(1).values[0]
        rsi = self.get_rsi_signal(26).tail(5).values
        macd = self.get_macd_signal(12, 26, 9).tail(5).values

        adx = self.get_adx_signal(14).tail(1).values[0]

        buy = self.buy_stategy(current_price, self.predict_price, ema, bb_upper, bb_lower, rsi, macd, adx)
        sell = self.sell_stategy(current_price, self.predict_price, ema, bb_upper, bb_lower, rsi, macd, adx)
        if buy == sell:
            return 'hold'
        elif buy:
            return 'buy'
        elif sell:
            return 'sell'

    def get_ema_signal(self, window):
        return ta.trend.ema_indicator(self.df['Close'], window)

    def get_bb_signal(self, window):
        bb_h = ta.volatility.bollinger_hband(self.df['Close'], window)
        bb_l = ta.volatility.bollinger_lband(self.df['Close'], window)
        return bb_h, bb_l

    def get_rsi_signal(self, window):
        return ta.momentum.rsi(self.df['Close'], window)

    def get_macd_signal(self, window_fast=12, window_slow=26, window_sign=9):
        macd = ta.trend.macd_diff(self.df['Close'], window_slow, window_fast, window_sign)
        return macd

    def get_adx_signal(self, window):
        return ta.trend.adx(self.df['High'], self.df['Low'], self.df['Close'], window)


def run(json_file):
    # init_json_file = '/home/giang/aws_hackathon/data/sol_from_begin.json'
    technical_analysis = TechnicalAnalysis(json_file)
    result = technical_analysis.get_buy_sell_signal()
    logger.info('result: {}'.format(result))
    # print(result)
    plot_df = technical_analysis.get_df().tail(125)
    # fig, axes = mpf.plot(plot_df, type='candle', style='charles', volume=True, figratio=(20, 10), figscale=1.5,
    #                      returnfig=True)
    fig, ax, ax2 = plotCandlestick(plot_df, technical_analysis.predict_price_)
    # # fig = plotCandlestick(plot_df, technical_analysis.predict_price_)
    # # fig.show()
    # # pd save csv
    # predict = pd.DataFrame(technical_analysis.predict_price_, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    # # last timestamp df in index
    # last_timestamp = plot_df.index[-1]
    # # add timestamp to predict_df by increase seq with step = +4h
    # predict['Timestamp'] = pd.date_range(start=last_timestamp + timedelta(hours=4), periods=len(predict), freq='4H')
    # # set timestamp to index
    # # predict.set_index('Timestamp', inplace=True)
    # # print(plot_df.head())
    # # add the first row to predict_df by index last_timestamp with value of df last row in df
    # _tail_df = plot_df.tail(1)
    # # convert index to "Timestamp" column
    #
    # _tail_df.reset_index(inplace=True)
    # # rename column "index" to "Timestamp"
    # _tail_df.rename(columns={'index': 'Timestamp'}, inplace=True)
    # # print(plot_df.head())
    # # _tail_df.drop(columns=['Timestamp'], inplace=True)
    # predict = pd.concat([_tail_df, predict])
    #
    # plot_df.to_csv(os.path.join(root, 'plot_df.csv'))
    # predict.to_csv(os.path.join(root, 'predict.csv'), index=False)
    #
    # with open(signal_file, 'a+') as f:
    #     if result == "buy":
    #         ls = [result, plot_df.index[-1],  plot_df['Low'].iloc[-1]]
    #         f.write('|'.join([str(i) for i in ls]) + '\n')
    #     elif result == "sell":
    #         ls = [result, plot_df.index[-1],  plot_df['High'].iloc[-1]]
    #         f.write('|'.join([str(i) for i in ls]) + '\n')


    # fig.show()
    a = [max(p) for p in zip(*technical_analysis.predict_price_)][3]
    b = [min(p) for p in zip(*technical_analysis.predict_price_)][3]
    # _min = plot_df['Low'].min()
    # _max = plot_df['High'].max()
    _min = min(plot_df['Low'].min(), b)
    _max = max(plot_df['High'].max(), a)
    x_last = plot_df.index[-1]
    if result == 'buy':
        last_low = plot_df['Low'].iloc[-1]
        ax2.annotate('Buy', xy=(x_last, last_low), xytext=(0, -20), textcoords='offset points', ha='center',
                     va='bottom', fontsize=5, arrowprops=dict(arrowstyle='->', lw=1, color='green'))
    elif result == 'sell':
        last_high = plot_df['High'].iloc[-1]
        ax2.annotate('Sell', xy=(x_last, last_high), xytext=(0, 20), textcoords='offset points', ha='center',
                     va='bottom', fontsize=5, arrowprops=dict(arrowstyle='->', lw=1, color='red'))

    _range = _max - _min
    min_low = _min - _range * 0.25
    max_low = _max + _range * 0.25
    max_high = plot_df['High'].max() + 10
    ax2.set_ylim(min_low, max_low)
    fig.autofmt_xdate()
    # return fig
    logger.info('save figure')
    fig.savefig('/home/giang/aws_hackathon/assets/figures/candles.png', dpi=150)


if __name__ == "__main__":
    run('/home/giang/aws_hackathon/data/sol_from_begin.json')
