"""Test SIMPLEST case: single fact, single unbind."""

import jax
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

# Create model
model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 4)

def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

role1 = rand_vec(keys[0])
role2 = rand_vec(keys[1])
alice = rand_vec(keys[2])
bob = rand_vec(keys[3])

print("=== TEST 1: Basic bind/unbind ===")
# Test: R1 ⊗ alice, then unbind R1 to get alice back
bound = model.opset.bind(role1, alice)
role1_inv = model.opset.inverse(role1)
recovered = model.opset.bind(bound, role1_inv)
sim = cosine_similarity(recovered, alice)
print(f"Encode R1⊗alice, unbind R1, check similarity to alice: {sim:.4f}")
print(f"Expected: >0.8, Actual: {sim:.4f} - {'PASS' if sim > 0.8 else 'FAIL'}\n")

print("=== TEST 2: Bundle then unbind ===")
# Test: (R1⊗alice + R2⊗bob), unbind R1 to get alice
rf1 = model.opset.bind(role1, alice)
rf2 = model.opset.bind(role2, bob)
bundle = model.opset.bundle(rf1, rf2)
unbind_r1 = model.opset.bind(bundle, role1_inv)
sim_alice = cosine_similarity(unbind_r1, alice)
sim_bob = cosine_similarity(unbind_r1, bob)
print(f"Encode (R1⊗alice + R2⊗bob), unbind R1:")
print(f"  Similarity to alice: {sim_alice:.4f}")
print(f"  Similarity to bob:   {sim_bob:.4f}")
print(f"Expected: alice >0.6, bob ~0.0")
print(f"Actual: {'PASS' if sim_alice > 0.6 else 'FAIL'}\n")

print("=== TEST 3: Query pattern ===")
# Encode: R1⊗alice + R2⊗bob
# Query: R1⊗alice
# Unbind query, then unbind R2 to get bob
fact = model.opset.bundle(
    model.opset.bind(role1, alice),
    model.opset.bind(role2, bob)
)
query = model.opset.bind(role1, alice)
query_inv = model.opset.inverse(query)
unbind_query = model.opset.bind(fact, query_inv)

role2_inv = model.opset.inverse(role2)
entity = model.opset.bind(unbind_query, role2_inv)

sim_bob = cosine_similarity(entity, bob)
sim_alice = cosine_similarity(entity, alice)
print(f"Encode (R1⊗alice + R2⊗bob), query with R1⊗alice, unbind R2:")
print(f"  Similarity to bob:   {sim_bob:.4f}")
print(f"  Similarity to alice: {sim_alice:.4f}")
print(f"Expected: bob >0.6, alice ~0.0")
print(f"Actual: {'PASS' if sim_bob > 0.6 else 'FAIL'}\n")
