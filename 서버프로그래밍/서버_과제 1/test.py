def char_frequency(str1):
    freq = {}
    for char in str1:
        freq[char] = freq.get(char, 0) + 1
    return freq

print(char_frequency('google.com'))