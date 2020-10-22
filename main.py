import sys
from math import gcd, sqrt
from collections import defaultdict, Counter
import os

def find_gcd(numbers):
    x = numbers[0]
    for i in range(1, len(numbers)):
        x = gcd(x,numbers[i])
    return x

def get_gcd_divisors(number):
    divisors = [1]
    for i in range(2, int(sqrt(number)+1)):
        if number%i == 0:
            divisors.append(i)
    return divisors

def kasiski(file_name):
    repeating_graphs_dict = defaultdict(list)
    with open(file_name) as f_read:
        cipher = f_read.read()
        for j in [2, 3, 4, 5, 6]:
            for i in range(len(cipher) - j + 1):
                # consider j = [2,3,4,5,6] for now
                if cipher[i:i + j] in repeating_graphs_dict:
                    repeating_graphs_dict[cipher[i:i + j]].append(i)
                else:
                    repeating_graphs_dict[cipher[i:i + j]] = [i]

    # print(repeating_graphs_dict)
    repeating_patterns = {}
    for k, v in repeating_graphs_dict.items():
        if len(v) >= 2:
            repeating_patterns[k] = v

    # print(repeating_patterns)
    # calculating key distance
    pattern_distance = {}
    for k, v in repeating_patterns.items():
        distances = []
        for i in range(len(v) - 1):
            distances = distances + [v[j] - v[i] for j in range(i + 1, len(v))]
        pattern_distance[k] = distances

    # print(pattern_distance)
    pattern_gcd = {}
    for k, v in pattern_distance.items():
        pattern_gcd[k] = find_gcd(v)
    # print(pattern_gcd)

    # get divisors from gcd, and choose the one with maximum count and one before that
    pattern_gcd_divisor = {}
    for k, v in pattern_gcd.items():
        pattern_gcd_divisor[k] = get_gcd_divisors(v)

    key_size_counter = Counter()
    for v in pattern_gcd_divisor.values():
        key_size_counter.update(v)
    # print(key_size_counter)
    return key_size_counter


def incident_coeff():
    pass



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_tests = 2  # sys.argv[1]
    for tn in range(n_tests):
        curr_dir = os.getcwd()
        file_name = os.path.join(curr_dir, f'tests/test{tn+1}')
        possible_key_size = kasiski(file_name)
        print(possible_key_size)
        # consider only keys of size more than 3 and keys with counter
        possible_keys = possible_key_size.most_common(7) # considering onl
        keys_to_try = []
        for k,v in possible_keys:
            if k ==1 or k == 2:
                continue
            else:
                keys_to_try.append(k)
        print(keys_to_try)