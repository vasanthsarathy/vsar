"""Test using circular shifts (permute) for positional encoding."""

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

print("=== Circular Shift (Permute) Encoding ===")
print("Idea: shift(entity, n) encodes entity at position n")
print("      shift(fact, -n) decodes position n")
print()

print("TEST 1: Basic shift and inverse shift")
# Shift alice by 100
shifted = model.opset.permute(alice, 100)
# Shift back by -100
recovered = model.opset.permute(shifted, -100)
sim = cosine_similarity(recovered, alice)
print(f"shift(alice, 100) then shift(_, -100): {sim:.4f}")
print(f"Expected: ~1.0 (perfect recovery)")
print(f"Result: {'PASS!' if sim > 0.99 else 'FAIL'}")
print()

print("TEST 2: Encode fact with shifts")
# Encode parent(alice, bob)
# Position 1: shift by 1
# Position 2: shift by 2
pos1_alice = model.opset.permute(alice, 1)
pos2_bob = model.opset.permute(bob, 2)
fact = model.opset.bundle(pos1_alice, pos2_bob)

# Decode position 2
decoded_pos2 = model.opset.permute(fact, -2)
sim_bob = cosine_similarity(decoded_pos2, bob)
sim_alice = cosine_similarity(decoded_pos2, alice)

print(f"Encode (shift(alice,1) + shift(bob,2)), decode with shift(-2):")
print(f"  Similarity to bob:   {sim_bob:.4f}")
print(f"  Similarity to alice: {sim_alice:.4f}")
print(f"Expected: bob >0.5, Result: {'PASS!' if sim_bob > 0.5 else 'FAIL'}")
print()

print("TEST 3: Full query pattern with shifts")
# parent(alice, bob) = shift(alice, 1) + shift(bob, 2)
fact1 = model.opset.bundle(
    model.opset.permute(alice, 1),
    model.opset.permute(bob, 2)
)

# parent(alice, carol) = shift(alice, 1) + shift(carol, 2)
fact2 = model.opset.bundle(
    model.opset.permute(alice, 1),
    model.opset.permute(carol, 2)
)

kb_bundle = model.opset.bundle(fact1, fact2)

# Query: parent(alice, X) - what's at position 2 when position 1 is alice?
# Subtract alice from position 1, then decode position 2

# Approach: kb - shift(alice, 1), then decode with shift(-2)
alice_at_pos1 = model.opset.permute(alice, 1)

# Can't directly subtract in bundle, but we can:
# 1. Decode pos 2 from both facts (they both have alice at pos1)
decoded_kb_pos2 = model.opset.permute(kb_bundle, -2)

sim_bob = cosine_similarity(decoded_kb_pos2, bob)
sim_carol = cosine_similarity(decoded_kb_pos2, carol)
sim_alice = cosine_similarity(decoded_kb_pos2, alice)

print(f"Simple decoding of position 2 from KB:")
print(f"  Similarity to bob:   {sim_bob:.4f}")
print(f"  Similarity to carol: {sim_carol:.4f}")
print(f"  Similarity to alice: {sim_alice:.4f}")
print(f"\nThis should show bob+carol bundled together")
print(f"Max similarity: {max(sim_bob, sim_carol):.4f}")

if max(sim_bob, sim_carol) > 0.4:
    print("\n*** PERMUTE ENCODING WORKS MUCH BETTER! ***")
    print("But we still need a way to filter by position 1 = alice")
    print("Maybe use binding IN ADDITION to shifts?")
    print("Or use a different strategy for queries...")
else:
    print("\nPermute encoding also problematic")
