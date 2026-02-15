from bs4 import BeautifulSoup
import requests
from pathlib import Path
import os
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import csv
import random
from collections import Counter

class reddit_crawler():
    def __init__(self, 
                 output_path:Path|str  = Path(__file__) / "reddit.csv", 
                 subreddit_list: list[str] = ["nbadiscussion", "PoliticalDiscussion", "CasualConversation",
                    "SubredditDrama", "movies", "StockMarket"]) -> None:
        self.output_path = output_path
        self.subreddit_list = subreddit_list
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "cookie": "loid=000000001csgjwg1mp.2.1731384251074.Z0FBQUFBQm5NdE83YTEtTEwyQ3ZqaHVGOTktNGh0OTRwQlBVZDRsM1NtQTRRd3puS3ZQNjNNS1ZKanJMV2dCNWZOWVhqTHpxTGg2NS1kUnR0R040ZE52eUl2TVhXT1FfeXFHcFdCejN6blVlcENWVlFQWUZOcGxjN0ZhcTd4Um1sb1h2dWJUTVo0ZEk; csv=2; edgebucket=lHfp2vxbsXOAK7sOYw; reddit_session=eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpsVFdYNlFVUEloWktaRG1rR0pVd1gvdWNFK01BSjBYRE12RU1kNzVxTXQ4IiwidHlwIjoiSldUIn0.eyJzdWIiOiJ0Ml8xY3NnandnMW1wIiwiZXhwIjoxNzg2NTI1MjY0Ljg0MjA2OCwiaWF0IjoxNzcwODg2ODY0Ljg0MjA2OCwianRpIjoiUndSVXlJV3p4dlEzT3FfNURiaUxnMnc3VGhEX0xBIiwiYXQiOjEsImNpZCI6ImNvb2tpZSIsImxjYSI6MTczMTM4NDI1MTA3NCwic2NwIjoiZUp5S2pnVUVBQURfX3dFVkFMayIsImZsbyI6NiwiYW1yIjpbInB3ZCJdfQ.YoUWOGWf1dcFaTUZiryTpu8KiHVe6UrBlkRjD1QRbkGICwnzxpJAO9g7QG2t9xdVyDv3niJVv-gK92EHI-abO6qbXjhalZUQ0Bge_L-Pwz2TlWUm9BFsAEAhxQflUSh4blT7wTx6IQB76t5qQQGi3Nzj-bGoZ4jCY7gI8oqNcvT89yrqirzXSaMPagDkzCw_WNHviyhSZ3CzLbOCh1pW6ZKZ8c2Bp6KDYKcIgVtSn1-cYMecDszGH20u_3Wqd4WErmGLwqqh3SVX7BkpK7xstePN17KSh9kKy7yJVMYKClhUVhXMkzMwNnwBfv7Uj3tXLDIQe3A2AJ1AZDYeWD5Z3g; prefers_reduced_motion=false; prefers_reduced_motion_sync=true; is_video_autoplay_disabled=false; csrf_token=99e43581a283c5c93ca2c8fcdc540dcf; token_v2=eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1Njp2am9FL2w2SkFOeWxWK2VPcG9ySWRzcm5sVzRSTXgzMVVQaS8rQkNnUkdZIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNzcwOTc5NDA4LjcyMzE3LCJpYXQiOjE3NzA4OTMwMDguNzIzMTcsImp0aSI6IjdUZXU2SnBQSGM0VFE0d1YzUTZTcElZWi1wWEU0dyIsImNpZCI6IjBSLVdBTWh1b28tTXlRIiwibGlkIjoidDJfMWNzZ2p3ZzFtcCIsImFpZCI6InQyXzFjc2dqd2cxbXAiLCJhdCI6MSwibGNhIjoxNzMxMzg0MjUxMDc0LCJzY3AiOiJlSnhra2RHT3REQUloZC1GYTVfZ2Y1VV9tMDF0Y1lhc0xRYW9rM243RFZvY2s3MDdjRDRwSFA5REtvcUZEQ1pYZ3FuQUJGZ1RyVERCUnVUOW5MbTNnMmlOZTh0WXNabkNCRm13RkRya21MR3NpUVFtZUpJYXl4c21vSUxOeUZ5dXRHTk5MVDBRSnFoY01yZUZIcGMyb2JrYmk1NmRHRlc1ckR5b3NWZmwwdGpHRkxZbnhqY2JxdzJwdUM2bk1rbkxRdmtzWHZUak45VzM5dm16X1NhMEo4T0txdW1CM2hsSkNHNHNmcGltM2Q5VGs1NnRDeGExOTNxUTJ1ZDYzSzU5MWl3ME83ZWY2X2xySXhtWFkyaC1KdnQzMXktaEE0ODhMelBxQUVhczRVY1pkbVFkX2xVSFVMbWdKR01KNHRNSTVNcmwyMzhKdG12VHY4YnRFejk4TS1LbU5feldETlJ6Q2VMUXBfSDFHd0FBX184UTFlVFIiLCJyY2lkIjoicnk1ZzdQNmhsZzItMVczaEZKTWNuYThnc0tmMFAtZHR4TmFERmhncnl4RSIsImZsbyI6Mn0.be9vAfuIt2gfG7PJERcFOWBkInSAqzE0vA1GvM9nrvZVlRsZCiCslWAEN6GfYrbyO_lSDQU9uFRwgbd2eyd0XMEAzCiFMrqHreLhROtsHdUB7U4PdFsBtiLK40zQ_Q6wEkoTlcaysjPaSmTqQkqjykKPEMwFGzaG9miDBjFWGbKHOddz5GImK6SVdwXJLY5lkp7nXZVl2-3vfd2YiFktKOV7ZaNbZwmrAHksz6vHugInRoUqOqn49drciexq0tdowqB8fLEudmkTsrZKMArsu38WNGvboK__7vYWiltyY3GTWeZ1moEqXKhOCimUJDhhRxFEjOq9SJvCWMRPrbeQiQ; t2_1csgjwg1mp_recentclicks3=t3_1gdx19g%2Ct3_1qspojd; pc=qz; session_tracker=flidibkobppgrerffa.0.1770934533584.Z0FBQUFBQnBqbEVGcjFZLVY0ZGZSSkw5eWtlVHRxWVNKS0JZVjhpdEVKbFZjS2l2ZHJSY1haM0t1clhRLWt1clZSYlVJeE9fWEgzRkR3VFJMbnp5VlpyVEhvTzhUYnhxZkR1alljZ2dzY2JLZ2tLRnktYkJCMWpfLURnUTZkR3p4ZmZ4TnpJVERQVG0",
            "priority": "u=0, i",
            "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
        }

        session = requests.Session()
        session.headers.update(self.headers)
        retry = Retry(total=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)

        self.session = session
        self.cnt = 0

    def status_condition(self, url: str):
        try:
            response = self.session.get(url)
            return response
        except Exception as e:
            print(f"Encounter error while sending request to {url}\nError: {e}")
        return None

    def count_collected_docs(self):
        file = self.output_path
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                row_count = len(reader)

                if row_count > 1:
                    last_subreddit = reader[-1][0]
                    last_name = f"t3_{reader[-1][1]}"
                    print(f"Resuming from: {last_subreddit}, last_name: {last_name}")
                else:
                    last_subreddit = None
                    last_name = None

            print("total:", row_count - 1)
            self.cnt = row_count - 1
            return (last_subreddit, last_name)
        else:
            with open(file, "w", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["subreddit", "id", "title", "content", "author", "num_comments", "created_utc"])

            self.cnt = 0
            return (None, None)

    def count_unique_values(self, column_name: str):
        values = []

        with open(self.output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            if column_name not in reader.fieldnames:
                print(f"Error: Column '{column_name}' not found!")
                print(f"Available columns: {reader.fieldnames}")
                return None

            for row in reader:
                values.append(row[column_name])

        counter = Counter(values)

        print(f"\n=== Statistics for column '{column_name}' ===")
        print(f"Total rows: {len(values)}")
        print(f"Unique values: {len(counter)}")
        print(f"\nValue counts:")
        for value, count in counter.most_common():
            print(f"  {value}: {count}")

        return dict(counter)

    def subreddit_crawler(self, url: str, subreddit: str):
      with open(self.output_path, "a", newline="", encoding="utf-8") as csvfile:
          writer = csv.writer(csvfile)

          while self.cnt < 5001:
              res = self.status_condition(url)
              if not res:
                  print(f"Request failed while crawling {url}")
                  break

              if res.status_code != 200:
                  print(f"Error: {res.status_code}")
                  break

              print(f"Success crawl {url}...")
              data = res.json()
              children = data["data"]["children"]

              if not children:
                  print(f"No more posts in r/{subreddit}")
                  break

              post_meta = [
                  (i["data"]["name"],
                    i["data"]["subreddit"],
                    i["data"]["id"],
                    i["data"]["title"],
                    i["data"]["selftext"],
                    i["data"]["author"],
                    i["data"]["num_comments"],
                    i["data"]["created_utc"])
                  for i in children
              ]

              for row in post_meta:
                  if row[4].strip():
                      writer.writerow(list(row[1:]))
                      self.cnt += 1

              last_name = post_meta[-1][0]
              url = f"https://old.reddit.com/r/{subreddit}/.json?count=25&after={last_name}"

              time.sleep(random.randint(10, 16))
              print(f"Success crawl {self.cnt} docs...")

    def main(self):
    #   subreddits = ["nbadiscussion", "PoliticalDiscussion", "CasualConversation",
    #                 "SubredditDrama", "movies", "StockMarket"]
        subreddits = self.subreddit_list
        last_subreddit, last_name = self.count_collected_docs()

        if last_subreddit and last_name:
            url = f"https://old.reddit.com/r/{last_subreddit}/.json?count=25&after={last_name}"
            print(f"Resuming from {last_subreddit}...")
            self.subreddit_crawler(url, last_subreddit)

            if last_subreddit in subreddits:
                start_index = subreddits.index(last_subreddit) + 1
                if start_index < len(subreddits):
                    for sub in subreddits[start_index:]:
                        url = f"https://old.reddit.com/r/{sub}/.json"
                        print(f"Starting new subreddit: {sub}")
                        self.subreddit_crawler(url, sub)
        else:
            for sub in subreddits:
                url = f"https://old.reddit.com/r/{sub}/.json"
                print(f"Starting subreddit: {sub}")
                self.subreddit_crawler(url, sub)

        self.count_unique_values("subreddit")
        print(f"Finally crawled {self.cnt} docs")
      
if __name__ == "__main__":
    cralwer = reddit_crawler()
    cralwer.main()