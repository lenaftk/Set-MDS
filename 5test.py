my_sets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 20, 11, 5, 23, 3, 4, 7, 16, 13, 26, 12, 2, 6, 5, 9, 6, 16, 16, 23, 34, 16, 11, 17, 30, 7]
original_sets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 22, 12, 5, 5, 11, 5, 9, 17, 23, 2, 5, 6, 24, 13, 21, 3, 16, 22, 6, 12, 20, 22, 4, 15, 7]

def count_elements(lst):
    counts = {}
    for x in lst:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    print(counts)
    return counts


def find_different_values(dict1, dict2):
    # find key-value pairs in dict1 that are not in dict2
    diff1 = {k: dict1[k] for k in set(dict1) - set(dict2) if dict1[k] != dict2.get(k)}

    # find key-value pairs in dict2 that are not in dict1
    diff2 = {k: dict2[k] for k in set(dict2) - set(dict1) if dict2[k] != dict1.get(k)}

    # return the combined set of different key-value pairs
    return {**diff1, **diff2}

print(find_different_values(dict1,dict2))
