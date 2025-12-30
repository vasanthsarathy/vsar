"""Test using permutations for positional encoding instead of binding."""

import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 3)

def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

alice = rand_vec(keys[0])
bob = rand_vec(keys[1])
carol = rand_vec(keys[2])

print("=== Permutation-Based Positional Encoding ===")
print("Idea: permute(entity, n) encodes entity at position n")
print("      inverse_permute(fact, n) decodes position n")
print()

print("TEST 1: Can we invert permute?")
# Permute alice
permuted = model.opset.permute(alice)
# Try to recover by permuting back (inverse permute)
# Note: permute might not have a direct inverse, but permuting d-1 times = inverse for permutation of order d
# For most VSA permutations, permute(permute(...)) cycles back
recovered = model.opset.permute(permuted)  # Try permuting again
sim = cosine_similarity(recovered, alice)
print(f"permute(alice), then permute again: {sim:.4f}")

# Try permuting multiple times to find the cycle
for i in range(2, 10):
    temp = permuted
    for _ in range(i):
        temp = model.opset.permute(temp)
    s = cosine_similarity(temp, alice)
    if s > 0.9:
        print(f"Found inverse at {i+1} permutations: {s:.4f}")
        inverse_permutes_needed = i + 1
        break
else:
    print("No simple inverse found for permute operation")
    inverse_permutes_needed = None

print()

if inverse_permutes_needed:
    print(f"TEST 2: Encode positions using {inverse_permutes_needed}-cycle permutation")
    # Position 0: entity (no permute)
    # Position 1: permute(entity)
    # Position 2: permute(permute(entity))

    # Encode parent(alice, bob)
    pos0_alice = alice  # position 0
    pos1_bob = model.opset.permute(bob)  # position 1

    fact = model.opset.bundle(pos0_alice, pos1_bob)

    # Decode position 1: apply inverse permutation
    decoded_pos1 = fact
    for _ in range(inverse_permutes_needed - 1):
        decoded_pos1 = model.opset.permute(decoded_pos1)

    sim_bob = cosine_similarity(decoded_pos1, bob)
    sim_alice = cosine_similarity(decoded_pos1, alice)

    print(f"  Similarity to bob:   {sim_bob:.4f}")
    print(f"  Similarity to alice: {sim_alice:.4f}")
    print(f"  Result: {'PASS' if sim_bob > 0.5 else 'FAIL'}")
    print()

print("=== Checking permute implementation ===")
print("Permute might be a random permutation, not a shift")
print("Let's check if it's deterministic and what it does:")

# Check if permute is deterministic
p1 = model.opset.permute(alice)
p2 = model.opset.permute(alice)
print(f"permute(alice) called twice, similarity: {cosine_similarity(p1, p2):.4f}")
print(f"Result: {'Deterministic' if cosine_similarity(p1, p2) > 0.99 else 'Non-deterministic'}")
