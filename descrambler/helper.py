import sys
import time


class Helper:

    timestamps = []

    @staticmethod
    def get_certain_loading(current_progress: int, final_num: int, description: str):
        t = "█"
        a = "."

        sys.stdout.write(
            f"\r{description}[{t * current_progress}{a * (final_num - current_progress)}] | {current_progress}/{final_num}")
        sys.stdout.flush()

    @staticmethod
    def get_uncertain_loading(count: int, description: str):
        progress = ["-", "\\", "|", "/", "-", "\\", "|", "/"]
        sys.stdout.write(
            f"\r{description}{progress[count % len(progress)]}")
        sys.stdout.flush()

    @staticmethod
    def current_milli_time():
        return round(time.time() * 1000)

    def add_timestamp(self):
        self.timestamps.append(Helper.current_milli_time())

    def get_latest_timestamp_difference(self):
        return self.timestamps[len(self.timestamps)-1] - self.timestamps[len(self.timestamps)-2]
