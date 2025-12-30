"""Minimal test to check if vsax query pattern works."""

import jax
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

# Create model
model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 6)

# Helper to create random vector
def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

# Create vectors
predicate = rand_vec(keys[0])
role1 = rand_vec(keys[1])
role2 = rand_vec(keys[2])
alice = rand_vec(keys[3])
bob = rand_vec(keys[4])
carol = rand_vec(keys[5])

print("Creating facts...")
# Encode facts: parent(alice, bob) and parent(alice, carol)
rf1_1 = model.opset.bind(role1, alice)
rf1_2 = model.opset.bind(role2, bob)
bundled1 = model.opset.bundle(rf1_1, rf1_2)
fact1 = model.opset.bind(predicate, bundled1)

rf2_1 = model.opset.bind(role1, alice)
rf2_2 = model.opset.bind(role2, carol)
bundled2 = model.opset.bundle(rf2_1, rf2_2)
fact2 = model.opset.bind(predicate, bundled2)

# KB bundle
kb_bundle = model.opset.bundle(fact1, fact2)

print("Creating query: parent(alice, X)")
# Query: parent(alice, X)
query = model.opset.bind(predicate, model.opset.bind(role1, alice))

print("Unbinding...")
# Unbind query from KB: kb ⊗ query^(-1)
query_inv = model.opset.inverse(query)
unbind_query = model.opset.bind(kb_bundle, query_inv)

# Unbind role2 to get entity: result ⊗ role2^(-1)
role2_inv = model.opset.inverse(role2)
entity_vec = model.opset.bind(unbind_query, role2_inv)

# Check similarities
sim_bob = cosine_similarity(entity_vec, bob)
sim_carol = cosine_similarity(entity_vec, carol)
sim_alice = cosine_similarity(entity_vec, alice)

print(f"\n=== RESULTS ===")
print(f"Similarity to bob:   {sim_bob:.4f}")
print(f"Similarity to carol: {sim_carol:.4f}")
print(f"Similarity to alice: {sim_alice:.4f}")
print(f"\nExpected: bob and carol should have HIGH similarity (>0.6)")
print(f"Actual max: {max(sim_bob, sim_carol):.4f}")

if sim_bob > 0.6 or sim_carol > 0.6:
    print("\n✓ PASS: Query pattern works!")
else:
    print(f"\n✗ FAIL: Query pattern doesn't work - scores too low!")
    print(f"This explains why VSAR is returning ~0.5 scores for all queries.")
