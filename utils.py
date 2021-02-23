import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f'Time elapsed in seconds: {(end - start):.2f}')
    return wrapper
