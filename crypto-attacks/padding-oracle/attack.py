from decrypt import check_padding
from operator import xor

def attack(cipher_text):
    bytes_padded = 1

    for i in range(16 - 1, -1, -1):
        iv = cipher_text[:16]
        ct = cipher_text[16:32]
        modified_iv = cipher_text[:16]
        modified_iv[i] = iv[i] + 1
        tampered_ct = modified_iv + ct

        if check_padding(tampered_ct):
            bytes_padded = i + 1
            break

    only_message_size = (len(cipher_text) - 16)
    message_bits = [0] * only_message_size

    for i in range(2, len(cipher_text) // 16):
        prev_ct = cipher_text[16 * (i - 1):16 * i]
        curr_ct = cipher_text[16 * i:16 * (i + 1)]
        modified_ct = cipher_text[16 * (i - 1):16 * i]

        for j in range(0, 16):
            for k in range(0, j):
                temp = xor(message_bits[16 * (i - 1) + k], (j + 1))
                modified_ct[k] = xor(prev_ct[k], temp)

            for l in range(0, 256):
                modified_ct[j] = xor(prev_ct[j], l)
                modified = modified_ct + curr_ct

                if not check_padding(modified):
                    block = 16 * (i - 1) + j
                    message_bits[block] = xor(l, (j + 1))
                    break

    for j in range(bytes_padded, 16):
        prev_ct = cipher_text[:16]
        curr_ct = cipher_text[16:32]
        modified_ct = cipher_text[:16]
        m = xor(bytes_padded, (j + 1))

        for t in range(0, bytes_padded):
            modified_ct[t] = xor(prev_ct[t], m)

        for p in range(bytes_padded, j):
            temp = xor(prev_ct[p], message_bits[p])
            modified_ct[p] = xor(temp, (j + 1))

        for k in range(0, 256):
            modified_ct[j] = xor(prev_ct[j], k)
            modified = modified_ct + curr_ct

            if not check_padding(modified):
                message_bits[j] = xor(k, (j + 1))
                break

    plaintext = message_bits[bytes_padded:]
    return [(byte % 256) for byte in plaintext]
