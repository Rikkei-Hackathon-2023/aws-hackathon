import os
import json
from datetime import datetime, timedelta

with open("/home/giang/aws_hackathon/data/sol_from_begin_org.json", "r") as f:
    data = json.load(f)

end_date = "2023-7-8 00:00:00"

end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
end_date = int(end_date.timestamp() * 1000)

valid_data = []

for d in data:
    if len(d) == 0:
        continue
    if d[0][0] > end_date:
        break
    valid_data.append(d)

with open("/home/giang/aws_hackathon/data/sol_from_begin.json", "w") as f:
    json.dump(valid_data, f)

with open("/home/giang/aws_hackathon/dag/count.txt", "w") as f:
    f.write("0")
