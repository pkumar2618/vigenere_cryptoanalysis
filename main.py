import sys
from math import gcd, sqrt
from collections import defaultdict, Counter
import os
import numpy as np

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
    with open(file_name, 'r') as f_read:
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


def key_len_subs_ic(cipher_text_file, key_sizes, print_key = False):
    with open(cipher_text_file, 'r') as f_read:
        cipher = f_read.read()
        cipher_len = len(cipher)
        keys_sub_ics = {}
        for key_size in key_sizes:
            sub_strings_ic = {}
            for i in range(key_size):
                i_string = ''.join([cipher[j] for j in range(i, cipher_len, key_size)])
                sub_strings_ic[i] = index_coincidence(i_string)

            keys_sub_ics[key_size] = sub_strings_ic
        if print_key:
            keys = set()
            for k, v in keys_sub_ics.items():
                # average_ic = sum(v.values())/k
                for ic in v.values():
                    if ic >= 0.065 and ic < 0.07:
                        keys.add(k)
            return keys
        else:
            return keys_sub_ics


def mutual_ic(x_string, y_string, type='string'):
    if type == 'string':
        x_f = Counter(x_string)
        x_n = sum(x_f.values())

        y_f = Counter(y_string)
        y_n = sum(y_f.values())

    elif type == 'Counter':
        x_f = x_string
        x_n = sum(x_f.values())
        y_f = y_string
        y_n = sum(y_f.values())

    common_symbols = set(x_f.keys()).intersection(y_f.keys())
    MIC_numerator = 0
    for k in common_symbols:
        MIC_numerator  += x_f[k]*y_f[k]
    return MIC_numerator/(x_n*y_n)

class Symbol():
    def __init__(self):
        self.symbol_sift = {}
        self.sift_symbol = {}
        symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(len(symbols)):
            self.symbol_sift[symbols[i]] = i
            self.sift_symbol[i] = symbols[i]

    def to_sift(self, symbol):
        return self.symbol_sift[symbol]

    def sifted_symbol(self, symbol, sift):
        original_position = self.symbol_sift[symbol]
        sifted_position = (original_position + sift)%26
        return self.sift_symbol[sifted_position]


def find_key_text(cipher_text_file, key_len, print_key = False):
    with open(cipher_text_file, 'r') as f_read:
        cipher = f_read.read()
        cipher_len = len(cipher)
        subs_rel_sift = {}
        subs_0 = ''.join([cipher[j] for j in range(0, cipher_len, key_len)])
        for i in range(1, key_len):
            sub_i = ''.join([cipher[j] for j in range(i, cipher_len, key_len)])
            subs_rel_sift[i] = mic_rel_sift(subs_0, sub_i)

        print(subs_rel_sift)
        stop_words = ['WHAT', 'WHERE', 'HOWEVER', 'WERE', 'HAVE',
                      'DOES', 'BECAUSE', 'THUS', 'HENCE', 'THERE']
        # finding the actual key_cod
        symbols = Symbol()
        for i in range(26):
            key = [i]
            for k, v in subs_rel_sift.items():
                key.append((i + 26-v) % 26)
            decipher = vigenere_decipher(cipher, key)
            ic = index_coincidence(decipher)

            for word in stop_words:
                if word in decipher:
                    return ''.join([symbols.sifted_symbol('A', sift) for sift in key]), decipher

def vigenere_decipher(cipher, key):
    symbols = Symbol()
    plain_text = ''
    cipher_len = len(cipher)
    block_size = len(key)
    for i in range(0, cipher_len, block_size):
        block_text = cipher[i:i+block_size]
        sifted_block_text = ''.join([symbols.sifted_symbol(och, -sch) for och, sch in zip(block_text, key)])
        plain_text = plain_text + sifted_block_text
    return plain_text


def mic_rel_sift(x_string, y_string):
    x_f = Counter(x_string)
    x_n = sum(x_f.values())

    y_original = Counter(y_string)

    symbol = Symbol()
    mic_trend = []
    for sift in range(26):
        y_sifted = Counter()
        for k in y_original:
            y_sifted[symbol.sifted_symbol(k, sift)] = y_original[k]

        mic_y_sifted = mutual_ic(x_f, y_sifted, type='Counter')
        mic_trend.append(mic_y_sifted)

    # finding the rel sift
    sift = np.argmax(np.array(mic_trend))
    # sifts = [sift for sift, value in enumerate(mic_trend) if value >= 0.059 and value < 0.07]
    return sift

def index_coincidence(x_string):
    x_f = Counter(x_string)
    x_n = sum(x_f.values())
    IC_numerator = 0
    for k in x_f.keys():
        IC_numerator  += x_f[k]*(x_f[k]-1)
    return IC_numerator/(x_n*(x_n-1))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_tests = 2  # sys.argv[1]
    for tn in range(n_tests):
        curr_dir = os.getcwd()
        file_name = os.path.join(curr_dir, f'tests/test{tn+1}')
        possible_key_size = kasiski(file_name)
        # print(possible_key_size)
        # consider only keys of size more than 3 and keys with counter
        possible_keys = possible_key_size.most_common(7) # considering onl
        keys_to_try = []
        for k,v in possible_keys:
            if k ==1 or k == 2:
                continue
            else:
                keys_to_try.append(k)
        # print(keys_to_try)

        keys_len = key_len_subs_ic(file_name, keys_to_try, print_key=True)
        print(keys_len)
        print(find_key_text(file_name, keys_len.pop()))
        # print(mutual_ic("ABACA", "BABAA"))