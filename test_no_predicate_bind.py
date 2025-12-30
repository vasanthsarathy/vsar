"""Test query WITHOUT binding predicate - use predicate as partition key only."""

import jax
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

# Create model
model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 5)

# Helper to create random vector
def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

# Create vectors - NO predicate needed!
role1 = rand_vec(keys[0])
role2 = rand_vec(keys[1])
alice = rand_vec(keys[2])
bob = rand_vec(keys[3])
carol = rand_vec(keys[4])

print("=== TEST: No Predicate Binding ===")
print("Idea: Since KB is already partitioned by predicate, we don't need to bind it!")
print()

print("Creating facts WITHOUT predicate binding...")
# Encode facts: just role-filler pairs!
# parent(alice, bob) => R1 ⊗ alice + R2 ⊗ bob
fact1 = model.opset.bundle(
    model.opset.bind(role1, alice),
    model.opset.bind(role2, bob)
)

# parent(alice, carol) => R1 ⊗ alice + R2 ⊗ carol
fact2 = model.opset.bundle(
    model.opset.bind(role1, alice),
    model.opset.bind(role2, carol)
)

# KB bundle (would be stored under 'parent' key)
kb_bundle = model.opset.bundle(fact1, fact2)

print("Creating query: parent(alice, X)")
# Query: just the bound role-filler
# parent(alice, X) => R1 ⊗ alice
query = model.opset.bind(role1, alice)

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
    print("\n*** SUCCESS! This approach works! ***")
    print("Solution: Don't bind predicate - use it only as KB partition key")
else:
    print(f"\n*** Still broken - scores too low ***")
