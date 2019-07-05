

def kld_coef(i):
    import math
    return (math.tanh((i - 3500)/1000) + 1)/2 