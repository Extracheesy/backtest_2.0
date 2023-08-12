from datetime import datetime, timedelta
from datetime import date

class ActivityTracker():
    def __init__(self):
        self.iteration = 0
        self.total_iteration = 0
        self.iteration_remaining = 0
        self.start = datetime.now()
        self.current_timestamp = datetime.now()

    def set_total_iteration(self, total_iteration):
        self.total_iteration = total_iteration
        self.iteration_remaining = self.total_iteration

    def increment_tic(self):
        self.iteration += 1
        self.iteration_remaining = self.total_iteration - self.iteration

    def display_tracker(self):
        self.current_timestamp = datetime.now()
        self.elapsed_time = self.current_timestamp - self.start
        if self.iteration == 0:
            print("elapsed time: ", self.elapsed_time, " to be executed: ", self.total_iteration)
        else:
            print("elapsed time: ", self.elapsed_time, " iteration: ", self.iteration, " remaining: ", self.iteration_remaining)