from binance.client import Client
import json
import pandas as pd
import logging
from datetime import timedelta, datetime
import time
import argparse

api_key = 'zxzAmI9wSM5uQqvUdfcZ6SEqhSlZK5g1GLwMSfXzKcnNp8sUxgRR3ppB0XuW8WOc'
api_secret = 'ivxFyGNfubAyQWt29yr0x3k8l6OwMlJF0mI5jglykAoE1nG58M6nXk1cUwbnPBZK'
client = Client(api_key, api_secret)
original_data = "/home/giang/aws_hackathon/data/sol_from_begin.json"
save_data = "/home/giang/aws_hackathon/data/sol_from_begin.json"

# set logger file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('/home/giang/aws_hackathon/logs/crawler.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

start_time = "2023-7-8 00:00:00"
end_time = "2023-7-30 00:00:00"
start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
array_times = pd.date_range(start=start_time, end=end_time, freq="4H").strftime("%Y-%m-%d %H:%M:%S").tolist()
# get count from file
with open("/home/giang/aws_hackathon/dag/count.txt", "r") as f:
    count = int(f.read())

# update count
with open("/home/giang/aws_hackathon/dag/count.txt", "w") as f:
    f.write(str((count + 1) % len(array_times)))

start_time = array_times[count]


def get_price(time):
    """

    :param time: datetime in str
    :return:
    """
    global logger
    global original_data, start_time
    logger.info(f"Getting price at {start_time}")
    try:
        start_time = pd.to_datetime(start_time)
        end_time = start_time + timedelta(minutes=1)
        data = client.get_historical_klines("SOLUSDT", Client.KLINE_INTERVAL_4HOUR, start_str=str(start_time),
                                            end_str=str(end_time))
        data = [[arg] for arg in data]
        original_data_json = json.load(open(original_data, "r"))
        original_data_json.extend(data)
        json_object = json.dumps(original_data_json, indent=4)
        with open(save_data, "w") as outfile:
            outfile.write(json_object)
    except Exception as e:
        logger.error(f"Get price at {start_time} failed")
        logger.error(f"{e}")
        return

    logger.info(f"Get price at {start_time} successfully")
    logger.info(f"{data}")


def parse_args():
    parser = argparse.ArgumentParser(description='Crawl SOL price')
    parser.add_argument('--time', type=str, default="123",
                        help='time')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    get_price(args.time)
