"""Check what VSAX cosine_similarity actually returns for complex vectors."""

import jax
import jax.numpy as jnp
from vsax import sample_complex_random
from vsax.similarity import cosine_similarity


# Generate two random complex vectors
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)

vec1 = sample_complex_random(dim=8192, n=1, key=key1).squeeze()
vec2 = sample_complex_random(dim=8192, n=1, key=key2).squeeze()

# Normalize them
vec1_norm = vec1 / jnp.linalg.norm(vec1)
vec2_norm = vec2 / jnp.linalg.norm(vec2)

# Check self-similarity
self_sim = cosine_similarity(vec1_norm, vec1_norm)
print(f"Self-similarity: {float(self_sim):.6f}")

# Check cross-similarity
cross_sim = cosine_similarity(vec1_norm, vec2_norm)
print(f"Cross-similarity (random vectors): {float(cross_sim):.6f}")

# Check if it's signed or unsigned
print(f"\nIs cosine_similarity always positive? {cross_sim >= 0}")
print(f"Range appears to be: [0, 1]" if cross_sim >= 0 else "Range appears to be: [-1, 1]")

# What our backend does
backend_sim = (cross_sim + 1.0) / 2.0
print(f"\nAfter (sim + 1.0) / 2.0 transform: {float(backend_sim):.6f}")
print(f"This explains why random entities have ~0.50 similarity!")
