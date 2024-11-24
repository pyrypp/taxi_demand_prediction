import plot_generator
import schedule
import time
import gc
import os

os.environ['TZ'] = 'Europe/Helsinki'
time.tzset()

def job():
    plot_generator.main()
    gc.collect()

schedule.every(1).minute.do(job)

while True:
    schedule.run_pending()
    time.sleep(10)