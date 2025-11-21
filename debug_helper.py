
import os
def debug_task(pair):
    g, s, df = pair
    return os.getpid(), g, s, len(df)
