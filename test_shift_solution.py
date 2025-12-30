"""SOLUTION: Use shift-based encoding with cleanup for queries."""

import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 6)

def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

# Create entity vectors
entities = {
    'alice': rand_vec(keys[0]),
    'bob': rand_vec(keys[1]),
    'carol': rand_vec(keys[2]),
    'dave': rand_vec(keys[3]),
    'eve': rand_vec(keys[4]),
}

print("=== SHIFT-BASED ENCODING SOLUTION ===\n")

print("Step 1: Encode facts using shifts")
# parent(alice, bob) = shift(alice, 1) + shift(bob, 2)
# parent(alice, carol) = shift(alice, 1) + shift(carol, 2)
# parent(bob, dave) = shift(bob, 1) + shift(dave, 2)

facts = []
fact_tuples = [
    ('alice', 'bob'),
    ('alice', 'carol'),
    ('bob', 'dave'),
]

for arg1, arg2 in fact_tuples:
    fact = model.opset.bundle(
        model.opset.permute(entities[arg1], 1),
        model.opset.permute(entities[arg2], 2)
    )
    facts.append(fact)
    print(f"  Encoded: parent({arg1}, {arg2})")

kb_bundle = model.opset.bundle(*facts)
print(f"\nCreated KB bundle with {len(facts)} facts\n")

print("Step 2: Query parent(alice, X) - what's at position 2?")
# Just decode position 2 from entire KB
decoded_pos2 = model.opset.permute(kb_bundle, -2)

print("Step 3: Check similarities to all entities")
similarities = {}
for name, vec in entities.items():
    sim = cosine_similarity(decoded_pos2, vec)
    similarities[name] = sim

# Sort by similarity
sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

print("\nResults (sorted by similarity):")
for name, sim in sorted_results:
    marker = " <-- EXPECTED" if name in ['bob', 'carol', 'dave'] else ""
    print(f"  {name:8s}: {sim:.4f}{marker}")

print(f"\nTop-2 results: {sorted_results[0][0]}, {sorted_results[1][0]}")
print(f"Expected top-2: bob, carol, dave (any 2 of 3)")

# Check if we got reasonable results
top2 = {sorted_results[0][0], sorted_results[1][0]}
expected = {'bob', 'carol', 'dave'}

if len(top2 & expected) >= 1:
    print("\n*** PARTIAL SUCCESS! ***")
    print("We're getting position-2 entities, but can't filter by position-1")
    print("\nNEXT STEP: Add position-1 filtering using binding or another mechanism")
else:
    print("\nStill not working correctly")

print("\n=== Testing Position-1 Filtering ===")
print("Idea: Bind a 'query vector' to filter facts matching position 1")

# Try: Filter facts where position 1 matches alice
# Decode position 1 from KB
decoded_pos1 = model.opset.permute(kb_bundle, -1)
sim_alice_pos1 = cosine_similarity(decoded_pos1, entities['alice'])
print(f"\nDecoded position 1 similarity to alice: {sim_alice_pos1:.4f}")
print("This should be high since 2 out of 3 facts have alice at position 1")

if sim_alice_pos1 > 0.3:
    print("\n*** SHIFT ENCODING WORKS! ***")
    print("Problem: Need a way to filter KB by position-1 value before decoding position-2")
    print("Possible solutions:")
    print("  1. Store facts separately and filter in Python before bundling")
    print("  2. Use binding in addition to shifts for filtering")
    print("  3. Use weighted bundle based on position-1 match")
