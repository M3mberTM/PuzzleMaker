from debug.debug import Debug as Debug


class Logger:
    @staticmethod
    def info(msg: str):
        if Debug.DEBUG:
            print(msg)

    @staticmethod
    def error(msg: str):
        if Debug.DEBUG:
            print(f"\x1b[31m{msg}")