from datetime import datetime


class Logger:
    _instance = {}

    def __init__(self, filename):
        if not hasattr(self, "initialized"):
            self.filename = filename
            self.outfile = open(filename, "w")
            self.initialized = True

    def __new__(cls, filename):
        if filename not in cls._instance.keys():
            cls._instance[filename] = super(Logger, cls).__new__(cls)
        return cls._instance[filename]

    @classmethod
    def get_logger(cls, filename):
        return cls._instance[filename] if filename in cls._instance.keys() else None

    def print(self, message, severity="", silent = False, outs = True):
        severity = f"[{severity}] " if severity != "" else severity
        self.write_message(f"[{str(datetime.now())}]: {severity.upper()}{message}\n", silent = silent, outs = outs)

    def write_message(self, message, silent = False, outs = True):
        if not silent:
            print(message, end = "")
        if outs:
            self.outfile.writelines(message)

    def close(self):
        if self.outfile:
            self.outfile.close()
            self.outfile = None

        if self.filename in Logger._instance:
            del Logger._instance[self.filename]
