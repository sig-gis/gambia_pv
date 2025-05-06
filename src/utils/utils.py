import os
import glob
import re


def find_lowest_loss(directory):
    files = glob.glob(directory)
    loss_regex = re.compile(r'loss(0\.\d+)')
    min_loss = float('inf')
    best_file = None
    for f in files:
        match = loss_regex.search(os.path.basename(f))
        if match:
            loss = float(match.group(1))
            if loss < min_loss:
                min_loss = loss
                best_file = f

    return best_file