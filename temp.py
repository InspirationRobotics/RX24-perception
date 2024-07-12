import time

# print unix time every 0.1 second
while True:
    print(time.time(), end='\r')
    time.sleep(0.1)
