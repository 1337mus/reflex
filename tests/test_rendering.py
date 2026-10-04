from reflex_decisions import rendering
from reflex_decisions.schema import DecisionRequest, Option


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


class WhitespaceMergingTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if len(text) >= 2 and text[-2] == " " and text[-1] in "ABCDEFGHIJKLMNOP":
            return [ord(character) for character in text[:-2]] + [
                1000 + ord(text[-2]) + ord(text[-1])
            ]
        return [ord(character) for character in text]


def test_compiled_candidate_symbols_map_back_to_semantic_ids() -> None:
    request = DecisionRequest(
        context="A request",
        question="What should happen?",
        options=(Option(id="hold", label="Hold"), Option(id="close", label="Close")),
    )

    compiled = rendering.compile_request(request, CharacterTokenizer())

    assert compiled.symbol_to_option_id == (("A", "hold"), ("B", "close"))
    assert len(compiled.candidate_token_ids) == len(request.options)


def test_answer_suffix_keeps_symbols_out_of_whitespace_merging() -> None:
    request = DecisionRequest(
        context="A request",
        question="What should happen?",
        options=(Option(id="hold", label="Hold"), Option(id="close", label="Close")),
    )

    compiled = rendering.compile_request(request, WhitespaceMergingTokenizer())

    assert compiled.prompt.endswith("\nAnswer:\n")
    assert compiled.candidate_token_ids[0] != compiled.candidate_token_ids[1]
