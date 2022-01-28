import sys

MY_LOG_FILE = 'progress.csv'

def eprintf(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def fprintf(*args, **kwargs):
    with open(MY_LOG_FILE, 'a') as f:
        f.write(*args, **kwargs)
        f.write('\n')



