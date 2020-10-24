import sys
from math import gcd, sqrt
from collections import defaultdict, Counter
import os
import numpy as np
import json

english_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


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

    # calculating key distance
    pattern_distance = {}
    for k, v in repeating_patterns.items():
        distances = []
        for i in range(len(v) - 1):
            distances = distances + [v[j] - v[i] for j in range(i + 1, len(v))]
        pattern_distance[k] = distances

    json_item = {'repeating_pattern_distances': pattern_distance}
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
    # return key_size_counter
    possible_keys = key_size_counter.most_common(7)  # considering onl
    keys_len_to_try = []
    for k, v in possible_keys:
        if k == 1 or k == 2:
            continue
        else:
            keys_len_to_try.append(k)
    # print(keys_to_try)
    json_item['possible_key_lengths'] = keys_len_to_try
    return json_item

def find_gcd(numbers):
    x = numbers[0]
    for i in range(1, len(numbers)):
        x = gcd(x, numbers[i])
    return x


def get_gcd_divisors(number):
    divisors = [1]
    for i in range(2, int(sqrt(number) + 1)):
        if number % i == 0:
            divisors.append(i)
    return divisors


def key_len_subs_ic(cipher_text_file, key_sizes, print_key=False):
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

        json_item = {'key_len_and_correponding_ICs':keys_sub_ics}
        if print_key:
            keys = set()
            for k, v in keys_sub_ics.items():
                for ic in v.values():
                    if ic >= 0.065 and ic < 0.07:
                        keys.add(k)
            # return keys
            json_item['verified_key_lengths'] = list(keys)
            return json_item
        else:
            # return keys_sub_ics
            return json_item

def index_coincidence(x_string):
    x_f = Counter(x_string)
    x_n = sum(x_f.values())
    IC_numerator = 0
    for k in x_f.keys():
        IC_numerator += x_f[k] * (x_f[k] - 1)
    return IC_numerator / (x_n * (x_n - 1))


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

    sift = np.argwhere(mic_trend == np.max(mic_trend))
    json_item = {'rel_sift': sift.flatten().tolist(), 'MIC': np.array(mic_trend)[sift].tolist()}

    return json_item

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
        MIC_numerator += x_f[k] * y_f[k]
    return MIC_numerator / (x_n * y_n)


def find_key_text(cipher_text_file, key_len, print_key=False):
    with open(cipher_text_file, 'r') as f_read:
        cipher = f_read.read()
        cipher_len = len(cipher)
        subs_rel_sift = {}
        subs_0 = ''.join([cipher[j] for j in range(0, cipher_len, key_len)])
        for i in range(1, key_len):
            sub_i = ''.join([cipher[j] for j in range(i, cipher_len, key_len)])
            subs_rel_sift[i] = mic_rel_sift(subs_0, sub_i)

        # print(subs_rel_sift)
        json_list = []
        json_item = {'key_length': key_len, 'positional_rel_sift': subs_rel_sift}
        # stop_words = ['WHAT', 'WHERE', 'HOWEVER', 'WERE', 'HAVE',
        #               'DOES', 'BECAUSE', 'THUS', 'HENCE', 'THERE']

        english_char_freq = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153,
                             0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056,
                             0.02758, 0.00978, 0.02360, 0.00150, 0.01974, 0.00074]
        english_freq = {english_alphabets[i]: english_char_freq[i] for i in range(26)}
        # finding the actual key_cod
        symbols = Symbol()
        found_key_decipher = []
        for i in range(26):
            key = [[i]]
            for k, v in subs_rel_sift.items():
                key.append(list((i + 26 - np.array(v['rel_sift'])) % 26))
            key_decipher = vigenere_decipher(cipher, key)
            # ic = index_coincidence(decipher)

            key_decipher_freq = [(key, decipher, Counter(decipher)) for key, decipher in key_decipher]
            for key, decipher, decipher_freq in key_decipher_freq:
                n = sum(decipher_freq.values(), 0.0)
                for k, v in decipher_freq.items():
                    decipher_freq[k] /= n

                found_char_freq = 0
                for symbol in decipher_freq.keys():
                    if (decipher_freq[symbol] >= english_freq[symbol] - english_freq[symbol] * 0.50) and \
                            (decipher_freq[symbol] <= (english_freq[symbol] + english_freq[symbol] * 0.50)):
                        found_char_freq += 1

                if found_char_freq >= (len(decipher_freq.keys()) // 2):
                    found_key_decipher.append((''.join([symbols.sifted_symbol('A', sift) for sift in key]), decipher))
                else:
                    None

            # for word in stop_words:
            #     if word in decipher:
            #         return ''.join([symbols.sifted_symbol('A', sift) for sift in key]), decipher
        json_item['key_text_tuples'] = found_key_decipher
    return json_item


def vigenere_decipher(cipher, key):
    symbols = Symbol()
    cipher_len = len(cipher)
    block_size = len(key)
    keys = [[]]
    for k_sift in key:
        if len(k_sift) > 1:
            keys = [flatten(keys[:]) for i in range(len(k_sift))]
            for i in range(len(keys)):
                kj = k_sift[i % len(k_sift)]
                keys[i].append(kj)
        else:
            for i in range(len(keys)):
                keys[i].append(k_sift[0])

    key_text = []
    for key in keys:
        plain_text = ''
        for i in range(0, cipher_len, block_size):
            block_text = cipher[i:i + block_size]
            sifted_block_text = ''.join([symbols.sifted_symbol(och, -sch) for och, sch in zip(block_text, key)])
            plain_text = plain_text + sifted_block_text
        key_text.append((key, plain_text))
    return key_text


def flatten(key):
    flat_key = []
    for item in key:
        if isinstance(item, list):
            flat_key = flat_key + item
        elif isinstance(item, int):
            flat_key.append(item)
    return flat_key


class Symbol():
    def __init__(self):
        self.symbol_sift = {}
        self.sift_symbol = {}
        symbols = english_alphabets
        for i in range(len(symbols)):
            self.symbol_sift[symbols[i]] = i
            self.sift_symbol[i] = symbols[i]

    def to_sift(self, symbol):
        return self.symbol_sift[symbol]

    def sifted_symbol(self, symbol, sift):
        original_position = self.symbol_sift[symbol]
        sifted_position = (original_position + sift) % 26
        return self.sift_symbol[sifted_position]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_tests = 2  # sys.argv[1]
    for tn in range(0, n_tests):
        curr_dir = os.getcwd()
        file_name = os.path.join(curr_dir, f'tests/test{tn + 1}')
        json_item = kasiski(file_name)
        # print(possible_key_size)

        output_directory = os.path.join(curr_dir, f'crypt_analysis')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        kasiski_output_file = os.path.join(output_directory, f'test{tn+1}_kasiski')
        with open(kasiski_output_file, 'w') as f_write:
            f_write.write(json.dumps(json_item, indent=4))

        keys_to_try = json_item['possible_key_lengths']

        output_file_ic = os.path.join(output_directory, f'test{tn+1}_ic')
        ic_json_item = key_len_subs_ic(file_name, keys_to_try, print_key=True)
        with open(output_file_ic,'w') as f_write:
            f_write.write(json.dumps(ic_json_item, indent=4))

        keys_len = ic_json_item['verified_key_lengths']
        json_key_mic_text = find_key_text(file_name, keys_len.pop())
        json_key_mic = []
        json_key_text = []
        # for json_item in json_key_mic_text:
        json_key_mic.append({k:v for k, v in json_key_mic_text.items() if not k=='key_text_tuples'})
        json_key_text.append({k:v for k, v in json_key_mic_text.items() if k == 'key_text_tuples'})

        output_file_key_mic = os.path.join(output_directory, f'test{tn + 1}_key_mic')
        with open(output_file_key_mic,'w') as f_write:
            json_item_string = json.dumps(json_key_mic, indent=4)
            f_write.write(json_item_string)

        output_file_text = os.path.join(output_directory, f'text{tn+1}')
        with open(output_file_text,'w') as f_write:
            f_write.write(json.dumps(json_key_text, indent=4))
