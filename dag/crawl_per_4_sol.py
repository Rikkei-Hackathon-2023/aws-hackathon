from datetime import datetime
from airflow import DAG
from airflow.operators.bash import *
from datetime import datetime, date, timedelta
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('/home/giang/aws_hackathon/logs/scheduler.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

default_args = {'owner': 'airflow'}
python_executable = "/home/giang/miniconda/envs/gianglt/bin/python"  # modify this path to your python executable
root = "/home/giang/aws_hackathon/engine"

with DAG(
        dag_id='crawl_sol_per_4h',
        description='DAG crawl SOL per 4h',
        start_date=datetime(2023, 7, 20),
        # schedule_interval='0 */4 * * *' # Run every 4 hours
        # schedule_interval='*/1 * * * *',
        schedule_interval=timedelta(seconds=4),
) as dag:
    crawl_sol_per_4h = BashOperator(
        task_id='crawl_sol_per_4h',
        bash_command=f'{python_executable} {os.path.join(root, "crawler.py")}',
        dag=dag
    )

    predict_sol_per_4h = BashOperator(
        task_id='predict_sol_per_4h',
        bash_command=f'{python_executable} {os.path.join(root, "technical_analysis.py")}',
        dag=dag
    )

    crawl_sol_per_4h >> predict_sol_per_4h
