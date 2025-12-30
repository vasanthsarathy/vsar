"""Test query pattern using MAP backend (Multiply-Add-Permute) instead of FHRR."""

import jax
import jax.numpy as jnp
from vsax import create_map_model, sample_random, cosine_similarity

# Create MAP model instead of FHRR
model = create_map_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 5)

def rand_vec(key):
    # MAP uses real vectors, not complex
    return sample_random(dim=8192, n=1, key=key).squeeze(axis=0)

# Create vectors
role1 = rand_vec(keys[0])
role2 = rand_vec(keys[1])
alice = rand_vec(keys[2])
bob = rand_vec(keys[3])
carol = rand_vec(keys[4])

print("=== Testing MAP Backend (Geometric Algebra) ===\n")

# Test basic bind/unbind first
print("TEST 1: Basic bind/unbind with MAP")
bound = model.opset.bind(role1, alice)
role1_inv = model.opset.inverse(role1)
recovered = model.opset.bind(bound, role1_inv)
recovered_norm = recovered / jnp.linalg.norm(recovered)
sim = cosine_similarity(recovered_norm, alice)
print(f"Bind role1*alice, unbind role1, similarity: {sim:.4f}")
print(f"Expected: >0.8, Result: {'PASS' if sim > 0.8 else 'FAIL'}\n")

if sim < 0.8:
    print("MAP basic unbind also fails - trying full query anyway...\n")

# Try full query pattern
print("TEST 2: Full query pattern with MAP")
fact1 = model.opset.bundle(
    model.opset.bind(role1, alice),
    model.opset.bind(role2, bob)
)

fact2 = model.opset.bundle(
    model.opset.bind(role1, alice),
    model.opset.bind(role2, carol)
)

kb_bundle = model.opset.bundle(fact1, fact2)
query = model.opset.bind(role1, alice)

query_inv = model.opset.inverse(query)
unbind_query = model.opset.bind(kb_bundle, query_inv)
unbind_query_norm = unbind_query / jnp.linalg.norm(unbind_query)

role2_inv = model.opset.inverse(role2)
entity_vec = model.opset.bind(unbind_query_norm, role2_inv)
entity_vec_norm = entity_vec / jnp.linalg.norm(entity_vec)

sim_bob = cosine_similarity(entity_vec_norm, bob)
sim_carol = cosine_similarity(entity_vec_norm, carol)
sim_alice = cosine_similarity(entity_vec_norm, alice)

print(f"Query results:")
print(f"  Similarity to bob:   {sim_bob:.4f}")
print(f"  Similarity to carol: {sim_carol:.4f}")
print(f"  Similarity to alice: {sim_alice:.4f}")
print(f"\nExpected: bob/carol >0.5")
print(f"Result: {'PASS' if max(sim_bob, sim_carol) > 0.5 else 'FAIL'}")

if max(sim_bob, sim_carol) > 0.5:
    print("\n*** MAP BACKEND WORKS! ***")
else:
    print("\nMAP backend also fails - need different approach")
