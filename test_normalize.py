"""Test if normalization is breaking unbind."""

import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, sample_complex_random, cosine_similarity

model = create_fhrr_model(dim=8192)
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 2)

def rand_vec(key):
    return sample_complex_random(dim=8192, n=1, key=key).squeeze(axis=0)

role = rand_vec(keys[0])
entity = rand_vec(keys[1])

print("=== Test: Normalization Effect ===\n")

# Bind
bound = model.opset.bind(role, entity)
print(f"Bound vector norm: {jnp.linalg.norm(bound):.4f}")

# Unbind WITHOUT normalization
role_inv = model.opset.inverse(role)
recovered_raw = model.opset.bind(bound, role_inv)
print(f"Recovered (raw) norm: {jnp.linalg.norm(recovered_raw):.4f}")

# Normalize the recovered vector
recovered_norm = recovered_raw / jnp.linalg.norm(recovered_raw)
print(f"Recovered (normalized) norm: {jnp.linalg.norm(recovered_norm):.4f}")

# Check similarities
sim_raw = cosine_similarity(recovered_raw, entity)
sim_norm = cosine_similarity(recovered_norm, entity)

print(f"\nSimilarity to original entity:")
print(f"  Without normalization: {sim_raw:.4f}")
print(f"  With normalization:    {sim_norm:.4f}")
print(f"\nExpected: >0.8")
print(f"Result: {'PASS' if max(sim_raw, sim_norm) > 0.8 else 'FAIL'}")

# Also check the original entity's norm
print(f"\nOriginal entity norm: {jnp.linalg.norm(entity):.4f}")
