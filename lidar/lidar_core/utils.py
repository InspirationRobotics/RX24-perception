import time
from typing import List
from threading import Thread, Lock

class BackgroundThread:
    def __init__(self, callback, *, rate : int = 10, args = None):
        self.callback = callback
        self.rate = rate
        self.args = args
        self.running = False
        self.thread = Thread(target=self.run)

    def start(self):
        self.running = True
        self.thread.start()

    def run(self):
        while self.running:
            self.callback(self.args)
            time.sleep(1/self.rate)

    def stop(self):
        self.running = False
        self.thread.join()

    def __del__(self):
        self.stop()


class BackgroundThreads:

    def __init__(self):
        self.threads : List[BackgroundThread] = []
        self.lock = Lock()

    def add_thread(self, thread : BackgroundThread):
        with self.lock:
            self.threads.append(thread)

    def add_custom_thread(self, callback, *, rate : int = 10, args = None):
        with self.lock:
            self.threads.append(BackgroundThread(callback, rate=rate, args=args))

    def remove_thread(self, thread : BackgroundThread):
        with self.lock:
            self.threads.remove(thread)

    def start_all(self):
        with self.lock:
            for thread in self.threads:
                thread.start()

    def stop_all(self):
        with self.lock:
            for thread in self.threads:
                thread.stop()
    
    def __del__(self):
        self.stop_all()