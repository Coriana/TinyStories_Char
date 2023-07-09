import os
MAX_HISTORY_LENGTH = 512

class History:
    def __init__(self, file_path="files/past_history.txt", max_lines=25):
        self.file_path = file_path
        self.max_lines = max_lines
        self.lines = []
        self.length = 0
        self.direction = "Task: You are an AI called Emeldar, interact with your chat/followers and make them feel welcome."
        self.load_history()

    def add(self, line):
        line = line + '\n'
        line_length = len(line)
        while (self.length + line_length) >= (MAX_HISTORY_LENGTH - len(self.direction)) and self.length >= 1:
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length
        self.append_to_file(line)
        
    def addload(self, line):
        line = line + '\n'
        line_length = len(line)
        while (self.length + line_length) >= (MAX_HISTORY_LENGTH - len(self.direction)) and self.length >= 1:
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                lines = file.readlines()[-self.max_lines:]
                for line in lines:
                    self.addload(line.strip())

    def append_to_file(self, line):
        with open(self.file_path, "a") as file:
            try:
                file.write(line)
            except:
                pass
            
    def __str__(self):
       # history = "Task: " + self.direction + "\n" + "".join(self.lines)
        while self.length > (MAX_HISTORY_LENGTH - len(self.direction)) and self.length > 1:
            self.length -= len(self.lines.pop(0))
        history =  "".join(self.lines)
       # if len(history) > MAX_HISTORY_LENGTH:
       #     history = history[-MAX_HISTORY_LENGTH:]
        return history
        
class DonationHistory:
    def __init__(self, file_path="files/past_donations.txt", max_lines=25):
        self.file_path = file_path
        self.max_lines = max_lines
        self.lines = []
        self.length = 0
        self.load_history()

    def add(self, line):
        line = line + '\n'
        line_length = len(line)
        while (self.length + line_length) > MAX_HISTORY_LENGTH and self.length >= 2:
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length
        self.append_to_file(line)
        
    def addload(self, line):
        line = line + '\n'
        line_length = len(line)
        while (self.length + line_length) > MAX_HISTORY_LENGTH and self.length >= 2:
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                lines = file.readlines()[-self.max_lines:]
                for line in lines:
                    self.addload(line.strip())

    def append_to_file(self, line):
        with open(self.file_path, "a") as file:
            file.write(line)

    def __str__(self):
      #  history = "Task: " + self.direction + "\n" + "".join(self.lines)
        history = "".join(self.lines)
       # if len(history) > MAX_HISTORY_LENGTH:
       #     history = history[-MAX_HISTORY_LENGTH:]
        return history

class FollowHistory:
    def __init__(self, file_path="files/past_follows.txt", max_lines=25):
        self.file_path = file_path
        self.max_lines = max_lines
        self.lines = []
        self.length = 0
        self.load_history()

    def add(self, line):
        line = line + '\n'
        line_length = len(line)
        while (self.length + line_length) > MAX_HISTORY_LENGTH and self.length >= 2:
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length
        self.append_to_file(line)
        
    def addload(self, line):
        line = line + '\n'
        line_length = len(line)
        while (self.length + line_length) > MAX_HISTORY_LENGTH and self.length >= 2:
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                lines = file.readlines()[-self.max_lines:]
                for line in lines:
                    self.addload(line.strip())

    def append_to_file(self, line):
        with open(self.file_path, "a") as file:
            file.write(line)

    def __str__(self):
      #  history = "Task: " + self.direction + "\n" + "".join(self.lines)
        history = "".join(self.lines)
        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]
        return history
        