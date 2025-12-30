"""Test full query pattern using circular correlation for unbinding."""

import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 5)

def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

def unbind_correlation(bound, factor):
    """Unbind using circular correlation instead of inverse."""
    bound_fft = jnp.fft.fft(bound)
    factor_fft = jnp.fft.fft(factor)
    result_fft = bound_fft * jnp.conj(factor_fft)
    result = jnp.fft.ifft(result_fft)
    return result / jnp.linalg.norm(result)

# Create vectors
role1 = rand_vec(keys[0])
role2 = rand_vec(keys[1])
alice = rand_vec(keys[2])
bob = rand_vec(keys[3])
carol = rand_vec(keys[4])

print("=== Full Query Test with Circular Correlation ===\n")

# Encode facts: parent(alice, bob) and parent(alice, carol)
fact1 = model.opset.bundle(
    model.opset.bind(role1, alice),
    model.opset.bind(role2, bob)
)

fact2 = model.opset.bundle(
    model.opset.bind(role1, alice),
    model.opset.bind(role2, carol)
)

kb_bundle = model.opset.bundle(fact1, fact2)

# Query: parent(alice, X)
query = model.opset.bind(role1, alice)

print("Unbinding query from KB using correlation...")
unbind_query = unbind_correlation(kb_bundle, query)

print("Unbinding role2 to get entity...")
entity_vec = unbind_correlation(unbind_query, role2)

# Check similarities
sim_bob = cosine_similarity(entity_vec, bob)
sim_carol = cosine_similarity(entity_vec, carol)
sim_alice = cosine_similarity(entity_vec, alice)

print(f"\n=== RESULTS ===")
print(f"Similarity to bob:   {sim_bob:.4f}")
print(f"Similarity to carol: {sim_carol:.4f}")
print(f"Similarity to alice: {sim_alice:.4f}")
print(f"\nExpected: bob and carol should be >0.5")
print(f"Actual max: {max(sim_bob, sim_carol):.4f}")

if sim_bob > 0.5 or sim_carol > 0.5:
    print("\n*** SUCCESS! ***")
    print("SOLUTION: Replace vsax inverse() with circular correlation for FHRR unbinding!")
    print("Implementation: unbind(a, b) = IFFT(FFT(a) * conj(FFT(b)))")
else:
    print(f"\nStill too low - need >0.5")
