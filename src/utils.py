def pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value] * (max_len - len(seq))
