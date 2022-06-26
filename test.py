import numpy as np

def convertStateToBoard(state, b=3):
        if state == 0:
            return  np.zeros((b, b))
        digits = []
        while len(digits) < b*b:
            digits.append(int(state % b))
            state //= b
        digits = np.array(digits)
        # curr_len = len(digits)
        # for i in range(b*b - curr_len):
        #     np.insert(digits,0,0)
        return digits.reshape(b, b)

print(convertStateToBoard(4031))