from reflex_decisions import schema


def test_request_hash_is_independent_of_option_order() -> None:
    option = schema.Option
    request_type = schema.DecisionRequest
    first = request_type(
        context="A support conversation",
        question="Which category fits?",
        options=(option(id="billing", label="Billing"), option(id="access", label="Access")),
    )
    reordered = request_type(
        context="A support conversation",
        question="Which category fits?",
        options=(option(id="access", label="Access"), option(id="billing", label="Billing")),
    )

    assert first.request_hash == reordered.request_hash
    assert first.schema_hash == reordered.schema_hash
