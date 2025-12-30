"""Test FHRR unbinding using circular correlation instead of inverse."""

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

print("=== FHRR Unbinding with Circular Correlation ===\n")

# Bind using vsax
bound = model.opset.bind(role, entity)
print(f"Bound using vsax bind")

# Unbind using circular correlation
# For complex vectors: correlation is FFT(a) * conj(FFT(b))
# Which is same as: IFFT(FFT(bound) * conj(FFT(role)))

bound_fft = jnp.fft.fft(bound)
role_fft = jnp.fft.fft(role)

# Circular correlation: multiply by conjugate in frequency domain
result_fft = bound_fft * jnp.conj(role_fft)
recovered = jnp.fft.ifft(result_fft)

# Normalize
recovered_norm = recovered / jnp.linalg.norm(recovered)

# Check similarity
sim = cosine_similarity(recovered_norm, entity)

print(f"Recovered using circular correlation")
print(f"Similarity to original entity: {sim:.4f}")
print(f"Expected: >0.8")
print(f"Result: {'PASS !!!!' if sim > 0.8 else 'FAIL'}")

# Compare with inverse method
print(f"\n=== Comparison with inverse method ===")
role_inv = model.opset.inverse(role)
recovered_inv = model.opset.bind(bound, role_inv)
recovered_inv_norm = recovered_inv / jnp.linalg.norm(recovered_inv)
sim_inv = cosine_similarity(recovered_inv_norm, entity)
print(f"Similarity using inverse: {sim_inv:.4f}")

print(f"\nConclusion:")
if sim > 0.8:
    print("*** SOLUTION FOUND: Use circular correlation for unbinding! ***")
    print("The vsax inverse() method doesn't implement proper FHRR unbinding.")
else:
    print("Still broken - need to investigate further")
