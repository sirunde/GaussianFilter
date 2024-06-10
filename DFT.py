import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import threading
import time

def nparray(a):
    return np.asarray(a)

def rawCalculation(s,a):
    F = 0.0
    for m in range(a,len(s)):
        F = F + (s[m] * np.exp(-1j * 2 * np.pi * s[0] * m / len(s)))
    return F

def dft(s, a= None):
    s = nparray(s)
    if a is None:
        a = len(s)

    if s == 1:
        return s

    else:
        FList = []
        for i in range(a):
            b = rawCalculation(s,a)
            FList.append(np.round(b,1))

    return np.asarray(FList)

def dft2d(s, a= None):
    s = nparray(s)

    if a is None:
        a = len(s)
        if a > 1:
            v = np.size(s[0])
        else:
            pass

    if v == 1:
        return a

    else:
        output = np.full_like(s, 0,dtype='complex')
        for i in range(a):
            for j in range(v):
                F = 0.0
                for m in range(a):
                    for n in range(v):
                        F = F + (s[m][n] * np.exp(-1j * 2 * np.pi * (m*i/a + n*j/v)))
                output[i][j] = np.round(F,1)


        f = np.empty(0)
        output = np.empty(0)
        output = np.append(output, dft(f))

        return output

if __name__ == "__main__":
    test = [[1,2,3,4],[5,6,7,8]]
    output = dft2d(test)
    print(f"output = {(output)}")
    print(" ")
    print((np.fft.fft2(test)))
