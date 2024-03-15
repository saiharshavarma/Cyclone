import time
import schedule
from django.core.management.base import BaseCommand
from real_time_prediction.views import predictRealTime

class Command(BaseCommand):
    help = 'Run real-time prediction every 30 seconds'

    def handle(self, *args, **options):
        # Schedule the task every 30 seconds
        schedule.every(30).seconds.do(predictRealTime)

        # Keep the process running to execute scheduled tasks
        while True:
            schedule.run_pending()
            time.sleep(1)