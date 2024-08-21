def attack(ciphertext, decrypt):
    """
    You are given a ciphertext(byte array) of size 48 on some random message of size 48 bytes.
    You are also given access to the decryption function which takes a ciphertext of size 48 and outputs 48 bytes message corresponding to the ciphertext
    Example Use: decrypt(ciphertext)

    NOTE:
        1. Ensure that ciphertext send as input to decrypt function is a byte array of size 48
        2. Only one query can be made to decrypt function

    TODO: Implement your code below
    """
    ct = bytearray([0] * 48)
    pt = decrypt(ct)

    m1 = pt[:16]
    m2 = pt[16:32]

    key = bytes(x ^ y for x, y in zip(m1, m2))

    return key
