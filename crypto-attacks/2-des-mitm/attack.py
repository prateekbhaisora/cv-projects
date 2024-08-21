from des import encrypt, decrypt, key_gen

def attack(message, ciphertext):
    possible_k1 = {}  # Storing intermediate ciphertexts and k1 values
    
    
    for key_index_k1 in range(2 ** 20):  # As keys are restricted to 20-bits
        k1 = key_gen(key_index_k1)
        intermediate_ct = encrypt(k1, message)
        possible_k1[intermediate_ct] = k1
    
    
    for key_index_k2 in range(2 ** 20):
        k2 = key_gen(key_index_k2)
        decrypted_intermediate = decrypt(k2, ciphertext)
        
        
        if decrypted_intermediate in possible_k1:
            return possible_k1[decrypted_intermediate], k2
            
    return None