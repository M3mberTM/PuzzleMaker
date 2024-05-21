import sys


class Helper:

    @staticmethod
    def update_loading_bar(current_progress: int, final_num: int, description: str):
        t = "█"
        a = "."

        sys.stdout.write(
            f"\r{description}[{t * current_progress}{a * (final_num - current_progress)}] | {current_progress}/{final_num}")
        sys.stdout.flush()
