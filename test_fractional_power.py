"""Test role-filler encoding using fractional powers instead of binding."""

import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 4)

def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

# Create a base vector for positions
base = rand_vec(keys[0])
alice = rand_vec(keys[1])
bob = rand_vec(keys[2])
carol = rand_vec(keys[3])

print("=== Fractional Power Encoding ===")
print("Idea: Use base^n to encode position n, then bind with entity")
print()

# Create position vectors using fractional power
pos1 = model.opset.fractional_power(base, 1.0)  # Position 1
pos2 = model.opset.fractional_power(base, 2.0)  # Position 2

print("TEST 1: Basic encode/decode with fractional power")
# Encode: alice at position 1
encoded = model.opset.bind(pos1, alice)

# Decode: unbind position 1
pos1_inv = model.opset.inverse(pos1)
decoded = model.opset.bind(encoded, pos1_inv)
decoded_norm = decoded / jnp.linalg.norm(decoded)

sim = cosine_similarity(decoded_norm, alice)
print(f"Encode alice at pos1, decode with pos1^-1: {sim:.4f}")
print(f"Expected: >0.7, Result: {'PASS' if sim > 0.7 else 'FAIL'}")
print()

print("TEST 2: Bundle with multiple positions")
# Encode: parent(alice, bob) = alice@pos1 + bob@pos2
fact = model.opset.bundle(
    model.opset.bind(pos1, alice),
    model.opset.bind(pos2, bob)
)

# Decode position 2
pos2_inv = model.opset.inverse(pos2)
decoded2 = model.opset.bind(fact, pos2_inv)
decoded2_norm = decoded2 / jnp.linalg.norm(decoded2)

sim_bob = cosine_similarity(decoded2_norm, bob)
sim_alice = cosine_similarity(decoded2_norm, alice)

print(f"Encode (alice@pos1 + bob@pos2), decode pos2:")
print(f"  Similarity to bob:   {sim_bob:.4f}")
print(f"  Similarity to alice: {sim_alice:.4f}")
print(f"Expected: bob >0.5, Result: {'PASS' if sim_bob > 0.5 else 'FAIL'}")
print()

print("TEST 3: Query pattern with fractional powers")
# Facts: parent(alice, bob) and parent(alice, carol)
fact1 = model.opset.bundle(
    model.opset.bind(pos1, alice),
    model.opset.bind(pos2, bob)
)

fact2 = model.opset.bundle(
    model.opset.bind(pos1, alice),
    model.opset.bind(pos2, carol)
)

kb_bundle = model.opset.bundle(fact1, fact2)

# Query: parent(alice, X) - what's at position 2 when position 1 is alice?
# Unbind alice from position 1
query = model.opset.bind(pos1, alice)
query_inv = model.opset.inverse(query)
unbind_query = model.opset.bind(kb_bundle, query_inv)

# Now unbind position 2
unbind_pos2 = model.opset.bind(unbind_query, pos2_inv)
unbind_pos2_norm = unbind_pos2 / jnp.linalg.norm(unbind_pos2)

sim_bob = cosine_similarity(unbind_pos2_norm, bob)
sim_carol = cosine_similarity(unbind_pos2_norm, carol)
sim_alice = cosine_similarity(unbind_pos2_norm, alice)

print(f"Query: parent(alice, X)")
print(f"  Similarity to bob:   {sim_bob:.4f}")
print(f"  Similarity to carol: {sim_carol:.4f}")
print(f"  Similarity to alice: {sim_alice:.4f}")
print(f"\nExpected: bob/carol >0.3")
print(f"Actual max: {max(sim_bob, sim_carol):.4f}")

if max(sim_bob, sim_carol) > 0.3:
    print("\n*** FRACTIONAL POWER APPROACH SHOWS PROMISE! ***")
else:
    print("\nStill broken with fractional powers")

print("\n=== Alternative: Use permute for positions ===")
print("Instead of base^n, use permute^n which is O(1) and might be more stable")
