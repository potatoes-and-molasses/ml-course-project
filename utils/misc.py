


def sentencify(inference_results, dct):
    
    sentences = [[] for i in inference_results[0]]  
    reverse_dict = {dct[j]:j for j in dct}
    for i in range(len(inference_results)):
        for j in range(len(inference_results[0])):
            sentences[j].append(reverse_dict[inference_results[i][j].item()])
    
    return sentences

def kld_coef(i):
    import math
    return (math.tanh((i - 3500)/1000) + 1)/2 