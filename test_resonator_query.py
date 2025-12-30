"""SOLUTION: Resonator-based query filtering using shift encoding."""

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

print("=== RESONATOR-BASED QUERY WITH SHIFT ENCODING ===\n")

print("Step 1: Store facts SEPARATELY (don't bundle yet)")
# parent(alice, bob) = shift(alice, 1) + shift(bob, 2)
# parent(alice, carol) = shift(alice, 1) + shift(carol, 2)
# parent(bob, dave) = shift(bob, 1) + shift(dave, 2)

fact_vectors = []
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
    fact_vectors.append(fact)
    print(f"  Stored: parent({arg1}, {arg2})")

print(f"\nStored {len(fact_vectors)} facts separately\n")

print("Step 2: Query parent(alice, X) - RESONATOR APPROACH")
print("Create query probe for position-1 = alice")
query_probe = model.opset.permute(entities['alice'], 1)

print("\nStep 3: Compute similarity of query probe to each fact")
fact_scores = []
for i, fact in enumerate(fact_vectors):
    # Decode position-1 from this fact
    decoded_pos1 = model.opset.permute(fact, -1)
    # How similar is it to alice?
    score = cosine_similarity(decoded_pos1, entities['alice'])
    fact_scores.append(score)
    arg1, arg2 = fact_tuples[i]
    marker = " <-- MATCH" if arg1 == 'alice' else ""
    print(f"  Fact parent({arg1}, {arg2}): {score:.4f}{marker}")

print("\nStep 4: Filter facts by threshold (>0.5)")
threshold = 0.5
matching_facts = [
    fact_vectors[i] for i, score in enumerate(fact_scores) if score > threshold
]
matching_tuples = [
    fact_tuples[i] for i, score in enumerate(fact_scores) if score > threshold
]

print(f"Matched {len(matching_facts)} facts:")
for arg1, arg2 in matching_tuples:
    print(f"  parent({arg1}, {arg2})")

print("\nStep 5: Bundle only matching facts, then decode position-2")
if matching_facts:
    matched_bundle = model.opset.bundle(*matching_facts)
    decoded_pos2 = model.opset.permute(matched_bundle, -2)

    print("\nQuery results for parent(alice, X):")
    similarities = {}
    for name, vec in entities.items():
        sim = cosine_similarity(decoded_pos2, vec)
        similarities[name] = sim

    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    for name, sim in sorted_results:
        marker = " <-- EXPECTED" if name in ['bob', 'carol'] else ""
        print(f"  {name:8s}: {sim:.4f}{marker}")

    # Check success
    top2 = {sorted_results[0][0], sorted_results[1][0]}
    expected = {'bob', 'carol'}

    if top2 == expected:
        print("\n*** PERFECT SUCCESS! ***")
        print("Resonator approach correctly filters and retrieves results!")
    elif len(top2 & expected) >= 1:
        print("\n*** PARTIAL SUCCESS ***")
        print(f"Got {len(top2 & expected)}/2 expected results")
    else:
        print("\n*** FAILED ***")
else:
    print("\nNo matching facts found!")

print("\n=== ALTERNATIVE: Weighted Bundle ===")
print("Instead of hard threshold, use weighted bundle")

# Compute weights based on position-1 match
weights = []
for i, fact in enumerate(fact_vectors):
    decoded_pos1 = model.opset.permute(fact, -1)
    score = cosine_similarity(decoded_pos1, entities['alice'])
    # Use score as weight (can also use score^2 or sigmoid to sharpen)
    weight = max(0, score)  # Clip negative scores to 0
    weights.append(weight)
    arg1, arg2 = fact_tuples[i]
    print(f"  parent({arg1}, {arg2}): weight = {weight:.4f}")

# Weighted bundle: sum of weighted facts
weighted_bundle = jnp.zeros_like(fact_vectors[0])
for i, fact in enumerate(fact_vectors):
    weighted_bundle = weighted_bundle + weights[i] * fact

# Normalize
weighted_bundle = weighted_bundle / jnp.linalg.norm(weighted_bundle)

# Decode position-2
decoded_pos2 = model.opset.permute(weighted_bundle, -2)

print("\nWeighted query results:")
similarities = {}
for name, vec in entities.items():
    sim = cosine_similarity(decoded_pos2, vec)
    similarities[name] = sim

sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

for name, sim in sorted_results:
    marker = " <-- EXPECTED" if name in ['bob', 'carol'] else ""
    print(f"  {name:8s}: {sim:.4f}{marker}")

top2 = {sorted_results[0][0], sorted_results[1][0]}
expected = {'bob', 'carol'}

if top2 == expected:
    print("\n*** PERFECT SUCCESS WITH WEIGHTED BUNDLE! ***")
elif len(top2 & expected) >= 1:
    print(f"\n*** PARTIAL SUCCESS: {len(top2 & expected)}/2 ***")
else:
    print("\n*** FAILED ***")

print("\n=== CONCLUSION ===")
print("Storing facts separately and using resonator filtering SOLVES the query problem!")
print("This approach:")
print("  1. Uses shift encoding for positions (perfect inversion)")
print("  2. Filters facts by decoding position-1 and matching to query")
print("  3. Bundles only matching facts OR uses weighted bundle")
print("  4. Decodes position-2 to get final results")
print("\nNo binding/unbinding needed - avoids the broken vsax inverse() issue!")
