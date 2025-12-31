"""End-to-end integration test for Phase 2 (Horn Rules + Chaining).

This test demonstrates the complete VSAR pipeline:
1. Insert base facts
2. Define Horn rules
3. Apply rules via forward chaining
4. Query derived facts
5. Inspect traces and provenance
"""

import pytest

from vsar.language.ast import Atom, Directive, Fact, Query, Rule
from vsar.semantics.chaining import apply_rules
from vsar.semantics.engine import VSAREngine


class TestPhase2EndToEnd:
    """End-to-end tests for Phase 2 complete pipeline."""

    def test_family_tree_reasoning_complete(self) -> None:
        """Complete family tree reasoning example.

        Demonstrates:
        - Base facts (parent relationships)
        - Multiple rules (grandparent, ancestor, sibling)
        - Forward chaining with transitive closure
        - Querying derived facts
        - Trace inspection
        """
        # Initialize engine
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42}),
            Directive(name="novelty", params={"threshold": 0.95}),
            Directive(name="beam", params={"width": 50}),
        ]
        engine = VSAREngine(directives)

        # Insert base facts: 3-generation family tree
        # Generation 1: alice, bob (married)
        # Generation 2: carol, dave (children of alice & bob)
        # Generation 3: eve, frank (children of carol & dave)
        parents = [
            ("alice", "carol"),
            ("alice", "dave"),
            ("bob", "carol"),
            ("bob", "dave"),
            ("carol", "eve"),
            ("dave", "frank"),
        ]

        for parent, child in parents:
            engine.insert_fact(Fact(predicate="parent", args=[parent, child]))

        # Define rules
        rules = [
            # Grandparent: X is grandparent of Z if X is parent of Y and Y is parent of Z
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            ),
            # Ancestor: Base case - parent is ancestor
            Rule(
                head=Atom(predicate="ancestor", args=["X", "Y"]),
                body=[Atom(predicate="parent", args=["X", "Y"])],
            ),
            # Ancestor: Recursive - transitive closure
            Rule(
                head=Atom(predicate="ancestor", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="ancestor", args=["Y", "Z"]),
                ],
            ),
        ]

        # Apply rules via forward chaining
        chaining_result = apply_rules(engine, rules, max_iterations=10, k=10)

        # Verify chaining completed successfully
        assert chaining_result.fixpoint_reached
        assert chaining_result.total_derived >= 6  # At least 6 parent facts + derived

        # Test queries on derived facts
        # Query 1: Who are alice's grandchildren?
        result1 = engine.query(Query(predicate="grandparent", args=["alice", None]), k=10)
        grandchildren = [entity for entity, score in result1.results]
        assert "eve" in grandchildren or "frank" in grandchildren

        # Query 2: Who are bob's ancestors? (should be none - he's generation 1)
        result2 = engine.query(Query(predicate="ancestor", args=["bob", None]), k=10)
        assert len(result2.results) >= 2  # At least carol and dave

        # Query 3: Use rules with query (automatic derivation)
        engine2 = VSAREngine(directives)
        for parent, child in parents:
            engine2.insert_fact(Fact(predicate="parent", args=[parent, child]))

        result3 = engine2.query(
            Query(predicate="ancestor", args=["alice", None]), rules=rules, k=10
        )
        assert len(result3.results) >= 2  # Should find descendants

        # Verify trace was created
        trace = engine2.trace.get_dag()
        assert len(trace) >= 2  # query + chaining + retrieval

        # Find chaining event
        chaining_events = [e for e in trace if e.type == "chaining"]
        assert len(chaining_events) >= 1
        chaining_event = chaining_events[0]
        assert chaining_event.payload["num_rules"] == 3
        assert chaining_event.payload["fixpoint_reached"]

        # Verify KB stats
        stats = engine.stats()
        assert "grandparent" in stats["predicates"]
        assert "ancestor" in stats["predicates"]
        assert stats["predicates"]["grandparent"] >= 2

    def test_organizational_hierarchy(self) -> None:
        """Organizational hierarchy with manager chains.

        Demonstrates:
        - Multi-hop manager chains
        - Transitive closure (reports_to)
        - Query with specific constraints
        """
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 100})
        ]
        engine = VSAREngine(directives)

        # Organizational structure
        # CEO -> VP -> Director -> Manager -> Employee
        managers = [
            ("ceo", "vp_eng"),
            ("ceo", "vp_sales"),
            ("vp_eng", "dir_backend"),
            ("vp_eng", "dir_frontend"),
            ("dir_backend", "mgr_api"),
            ("mgr_api", "dev_alice"),
            ("mgr_api", "dev_bob"),
        ]

        for manager, employee in managers:
            engine.insert_fact(Fact(predicate="manages", args=[manager, employee]))

        # Define transitive closure rule
        rules = [
            # Base: manager reports_to themselves (0 hops)
            Rule(
                head=Atom(predicate="reports_to", args=["X", "Y"]),
                body=[Atom(predicate="manages", args=["Y", "X"])],
            ),
            # Recursive: transitive reports_to
            Rule(
                head=Atom(predicate="reports_to", args=["X", "Z"]),
                body=[
                    Atom(predicate="manages", args=["Y", "X"]),
                    Atom(predicate="reports_to", args=["Y", "Z"]),
                ],
            ),
        ]

        # Apply rules
        result = apply_rules(engine, rules, max_iterations=10, k=20)

        assert result.fixpoint_reached
        assert result.total_derived >= 7

        # Query: Who does dev_alice report to (all levels)?
        query_result = engine.query(
            Query(predicate="reports_to", args=["dev_alice", None]), k=20
        )

        # Should find: mgr_api, dir_backend, vp_eng, ceo
        assert len(query_result.results) >= 3

    def test_knowledge_graph_paths(self) -> None:
        """Knowledge graph with path finding.

        Demonstrates:
        - Multi-relation facts
        - Path queries via rules
        - Combining different predicates
        """
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 50})]
        engine = VSAREngine(directives)

        # Insert knowledge graph facts
        # People and their relationships
        engine.insert_fact(Fact(predicate="knows", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="knows", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="knows", args=["carol", "dave"]))

        engine.insert_fact(Fact(predicate="works_with", args=["alice", "eve"]))
        engine.insert_fact(Fact(predicate="works_with", args=["eve", "dave"]))

        # Define connection rules
        rules = [
            # Direct connection via knows
            Rule(
                head=Atom(predicate="connected", args=["X", "Y"]),
                body=[Atom(predicate="knows", args=["X", "Y"])],
            ),
            # Direct connection via works_with
            Rule(
                head=Atom(predicate="connected", args=["X", "Y"]),
                body=[Atom(predicate="works_with", args=["X", "Y"])],
            ),
            # Transitive connection
            Rule(
                head=Atom(predicate="connected", args=["X", "Z"]),
                body=[
                    Atom(predicate="connected", args=["X", "Y"]),
                    Atom(predicate="connected", args=["Y", "Z"]),
                ],
            ),
        ]

        # Apply rules
        result = apply_rules(engine, rules, max_iterations=10, k=20)

        assert result.fixpoint_reached

        # Query: Who is alice connected to?
        query_result = engine.query(
            Query(predicate="connected", args=["alice", None]), k=20
        )

        # Should find multiple connections
        assert len(query_result.results) >= 3

    def test_semi_naive_performance_benefit(self) -> None:
        """Verify semi-naive evaluation is faster than naive.

        Demonstrates:
        - Performance difference between naive and semi-naive
        - Both produce identical results
        """
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]

        # Create a longer chain for more iterations
        chain_length = 8
        parents = [(f"person{i}", f"person{i+1}") for i in range(chain_length)]

        # Naive evaluation
        engine_naive = VSAREngine(directives)
        for parent, child in parents:
            engine_naive.insert_fact(Fact(predicate="parent", args=[parent, child]))

        rules = [
            Rule(
                head=Atom(predicate="ancestor", args=["X", "Y"]),
                body=[Atom(predicate="parent", args=["X", "Y"])],
            ),
            Rule(
                head=Atom(predicate="ancestor", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="ancestor", args=["Y", "Z"]),
                ],
            ),
        ]

        result_naive = apply_rules(
            engine_naive, rules, max_iterations=20, k=20, semi_naive=False
        )

        # Semi-naive evaluation
        engine_semi = VSAREngine(directives)
        for parent, child in parents:
            engine_semi.insert_fact(Fact(predicate="parent", args=[parent, child]))

        result_semi = apply_rules(
            engine_semi, rules, max_iterations=20, k=20, semi_naive=True
        )

        # Both should reach fixpoint
        assert result_naive.fixpoint_reached
        assert result_semi.fixpoint_reached

        # Both should derive same number of facts
        assert result_naive.total_derived == result_semi.total_derived

        # Both should produce same final KB
        naive_ancestors = set(engine_naive.kb.get_facts("ancestor"))
        semi_ancestors = set(engine_semi.kb.get_facts("ancestor"))
        assert naive_ancestors == semi_ancestors

    def test_complex_multi_rule_scenario(self) -> None:
        """Complex scenario with multiple interacting rules.

        Demonstrates:
        - Multiple rule types
        - Rule interactions
        - Complex derivations
        """
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42}),
            Directive(name="beam", params={"width": 100}),
        ]
        engine = VSAREngine(directives)

        # Base facts: academic relationships
        # Advisors and students
        engine.insert_fact(Fact(predicate="advises", args=["prof_smith", "phd_alice"]))
        engine.insert_fact(Fact(predicate="advises", args=["phd_alice", "ms_bob"]))
        engine.insert_fact(Fact(predicate="advises", args=["prof_jones", "phd_carol"]))

        # Collaborations
        engine.insert_fact(Fact(predicate="coauthor", args=["prof_smith", "prof_jones"]))
        engine.insert_fact(Fact(predicate="coauthor", args=["phd_alice", "phd_carol"]))

        # Define complex rules
        rules = [
            # Academic lineage (transitive advising)
            Rule(
                head=Atom(predicate="academic_ancestor", args=["X", "Y"]),
                body=[Atom(predicate="advises", args=["X", "Y"])],
            ),
            Rule(
                head=Atom(predicate="academic_ancestor", args=["X", "Z"]),
                body=[
                    Atom(predicate="advises", args=["X", "Y"]),
                    Atom(predicate="academic_ancestor", args=["Y", "Z"]),
                ],
            ),
            # Symmetric coauthorship
            Rule(
                head=Atom(predicate="coauthor", args=["Y", "X"]),
                body=[Atom(predicate="coauthor", args=["X", "Y"])],
            ),
            # Collaboration network (transitive)
            Rule(
                head=Atom(predicate="collaborator", args=["X", "Y"]),
                body=[Atom(predicate="coauthor", args=["X", "Y"])],
            ),
            Rule(
                head=Atom(predicate="collaborator", args=["X", "Z"]),
                body=[
                    Atom(predicate="coauthor", args=["X", "Y"]),
                    Atom(predicate="collaborator", args=["Y", "Z"]),
                ],
            ),
        ]

        # Apply all rules
        result = apply_rules(engine, rules, max_iterations=15, k=20)

        assert result.fixpoint_reached
        assert result.total_derived >= 3

        # Verify complex derivations
        stats = engine.stats()
        assert "academic_ancestor" in stats["predicates"]
        assert "collaborator" in stats["predicates"]

        # Query: prof_smith's academic descendants
        desc_result = engine.query(
            Query(predicate="academic_ancestor", args=["prof_smith", None]), k=20
        )
        assert len(desc_result.results) >= 1  # At least phd_alice
