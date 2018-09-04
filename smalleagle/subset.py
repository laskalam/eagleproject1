import numpy as np


def subset_sum(numbers, target, nsub=3, init=0, size=20):


    output = np.zeros(nsub)
    Output = []
    count = 0
    for i, n in enumerate(numbers):
        output[0] = n
        rest = target - n
        for m in numbers:
            if m > rest:
                continue
            else:
                output[1] = m
                output[2] = rest - m
            if count < init or count >= init+size:
                count += 1
                continue
            Output.append(np.copy(output))
            count += 1
    return Output
