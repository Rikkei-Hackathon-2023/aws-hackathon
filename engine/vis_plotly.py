import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

plot_df = "/home/giang/aws_hackathon/data/plot_df.csv"
predict_df = "/home/giang/aws_hackathon/data/predict.csv"

signal_file = "/home/giang/aws_hackathon/data/signal.txt"


def plotCandlestick(df, _predict, signal, alpha=0.3):
    """
    df schema: Timestamp(index), Open, High, Low, Close, Volume
    :param df: Historical data DataFrame
    :param _predict: Predicted data (same schema as df)
    signal: ["command_text", "x_cord", "y_cord"]
    :param alpha: Transparency of predicted bars
    :return: Plotly Figure object
    """

    # # Create a copy of the original DataFrame to avoid modifying the original data
    # df = df.copy()
    # _predict = _predict.copy()
    #
    # # Add timestamp to predict DataFrame by increasing seq with step = +4h
    # # _predict['Timestamp'] = pd.date_range(start=last_timestamp + timedelta(hours=4), periods=len(_predict), freq='4H')
    # # _predict.set_index('Timestamp', inplace=True)
    #
    # _predict = pd.DataFrame(_predict, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    # # last timestamp df in index
    # last_timestamp = df.index[-1]
    # _predict['Timestamp'] = pd.date_range(start=last_timestamp + timedelta(hours=4), periods=len(_predict), freq='4H')
    #
    # # Concatenate the last row of df to _predict to have a continuous plot
    # predict = pd.concat([df.tail(1), _predict])

    # Create the figure
    # fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.5, 0.5])

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 increasing_line_color='green',
                                 decreasing_line_color='red',
                                 showlegend=False,
                                 name='Candlestick'), row=1, col=1)
    fig.add_trace(go.Candlestick(x=predict['Timestamp'],
                                 open=predict['Open'],
                                 high=predict['High'],
                                 low=predict['Low'],
                                 close=predict['Close'],
                                 increasing_line_color='green',
                                 decreasing_line_color='red',
                                 showlegend=False,
                                 opacity=alpha,
                                 name='Predicted'), row=1, col=1)
    # set xaxis to timestamp
    # fig.update_xaxes(type='category', row=1, col=1)

    # plot signals text into row 1
    for sig in signal:
        fig.add_annotation(x=sig[1], y=sig[2], text=sig[0], showarrow=True, arrowhead=1, row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='orange', name='Volume', opacity=0.7), row=2, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1, secondary_y=True)
    fig.update_layout(xaxis_title='Timestamp', yaxis_title='Price', title='Candlestick Chart')
    return fig


if __name__ == "__main__":
    df = pd.read_csv(plot_df)
    predict = pd.read_csv(predict_df)
    print(predict.head())
    print(df.head())
    # set index to timestamp
    df.set_index('Timestamp', inplace=True)
    signals = open(signal_file, 'r').readlines()
    signals = [line.strip().split("|") for line in signals]
    fig = plotCandlestick(df, predict, signals)
    fig.show()
