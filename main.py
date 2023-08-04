import Levenshtein
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from statistics import mean

# changing seed runs a new instance of the cryptography. Keeping the same seed will not change random numbers for
# reproducibility
def run_crypto(seed):
    # Parameters for the Lorenz system
    sigma = 10
    rho = 28
    beta = 8 / 3

    # Consideration 2: Key Generation
    t_start = 0
    t_end = 10
    t_step = 0.01
    key = generate_key(seed, sigma, rho, beta, t_start, t_end, t_step)

    # Consideration 3: Encryption
    message = "Encryption"
    print("------------------------------")
    print("Original message:", message)
    encrypted_message = encrypt(message, key)
    print("Encrypted message:", encrypted_message)

    # Consideration 4: Decryption
    decrypted_message = decrypt(encrypted_message, key)
    print("Decrypted message:", decrypted_message)

    [similiarity, similiarity2] = analyze_encryption_effectiveness(message, encrypted_message, key)

    # want to plot the differences between closeness of message and seed

    return [similiarity, similiarity2]


# Consideration 1: Choice of Chaotic ODE Model
def lorenz_equations(t, u, sigma, rho, beta):
    x, y, z = u
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# Consideration 2: Key Generation
def generate_key(seed, sigma, rho, beta, t_start, t_end, t_step):
    np.random.seed(seed)
    x0 = np.random.uniform(-20, 20)
    y0 = np.random.uniform(-20, 20)
    z0 = np.random.uniform(0, 40)
    u0 = [x0, y0, z0]

    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end, t_step)

    sol = solve_ivp(
        lorenz_equations,
        t_span,
        u0,
        args=(sigma, rho, beta),
        dense_output=True
    )

    key = sol.sol(t_eval)
    return key


# Consideration 3: Encryption Algorithm
def encrypt(message, key):
    tempKey = np.array(key)
    encrypted = ""
    key_len = len(key[0])

    for i, char in enumerate(message):
        key_idx = i % key_len
        key_value = tempKey[:, key_idx]

        # Use bitwise XOR operation for efficiency
        encrypted_char = chr(ord(char) ^ int(key_value.sum()) % 256)
        encrypted += encrypted_char

    return encrypted


# Consideration 4: Decryption Algorithm
def decrypt(encrypted, key):
    decrypted = ""
    key_len = len(key[0])

    for i, char in enumerate(encrypted):
        key_idx = i % key_len
        key_value = key[:, key_idx]

        decrypted_char = chr(ord(char) ^ int(key_value.sum()) % 256)
        decrypted += decrypted_char

    return decrypted


# TEST METHODS AFTER THIS POINT

def analyze_encryption_effectiveness(message, encrypted_message, key):
    """
    Analyze the effectiveness of an encryption method using the considerations
    """
    # Check minimum number of single character edits required to get from the encrypted message to the original message

    divisor = float(max(len(message), len(encrypted_message)))
    numerator = float(Levenshtein.distance(message, encrypted_message))
    similarity = 1 - float(numerator / divisor)

    print(f"Similarity between the encrypted message and the original: {similarity}")

    # Check sensitivity to the key by slightly modifying the key and re-encrypting the message

    # The range will look like this: [-perturbation_range, perturbation_range]
    perturbation_range = 2

    # Modify the key using the perturbations
    perturbations = np.random.uniform(-perturbation_range, perturbation_range, size=(3, 1000))
    modified_key = key + perturbations

    modified_encrypted_message = encrypt(message, modified_key)

    divisor = float(max(len(encrypted_message), len(modified_encrypted_message)))
    numerator = float(Levenshtein.distance(encrypted_message, modified_encrypted_message))
    similarity2 = 1 - float(numerator / divisor)
    print(f"Modified encrypted message after slight perturbation is: {modified_encrypted_message}")
    print("Similarity between messages with slight perturbation is: {}".format(similarity2))
    return [similarity, similarity2]


def graphAnalysis():
    """
    Graphing representations of numerical analysis
    """

    # variable definitions
    ITERATIONS = 200
    dataSet = {}

    i = 0
    while i < ITERATIONS:
        dataSet.update({i: run_crypto(i)})
        i = i + 1

    # Extract the keys and values from the dictionary
    keys = list(dataSet.keys())
    values = list(dataSet.values())

    # Extract value 1 and value 2 separately
    value1 = [v[0] for v in values]
    value2 = [v[1] for v in values]

    # Generate the graph
    plt.plot(keys, value1, label='Original vs Encryption')
    plt.plot(keys, value2, label='Encryption vs Pertubation')
    plt.xlabel('Iterations')
    plt.ylabel('Similarity')
    plt.title('Encryption Differences per Iteration')
    plt.legend()
    plt.show()

    print("____________________________________________________________")
    print("Average similarity for original vs encryption", mean(value1))
    print("Average similarity for perturbation vs encryption", mean(value2))


graphAnalysis()
