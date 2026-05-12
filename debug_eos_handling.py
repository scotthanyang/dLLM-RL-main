from rl_eos import eos_response_metadata, force_after_first_eos_to_eos, truncate_text_at_first_eos


PAD = 0
EOS = 2
A = 10
B = 11
C = 12
D = 13


def training_mask(response_ids):
    labels, _ = force_after_first_eos_to_eos(response_ids, [EOS])
    p_mask = [True] * len(response_ids)
    return labels, p_mask


def run_case(name, response_ids, response_text):
    forced_ids, first_eos = force_after_first_eos_to_eos(response_ids, [EOS])
    metadata = eos_response_metadata(
        response_ids,
        eos_token_ids=[EOS],
        pad_token_ids=[PAD],
        text=response_text,
        eos_token_strings=["<EOS>"],
    )
    reward_text = truncate_text_at_first_eos(
        response_text, ["<EOS>"], include_eos=False
    )
    labels, p_mask = training_mask(response_ids)
    print(f"{name}:")
    print(f"  raw_ids={response_ids}")
    print(f"  forced_after_eos={forced_ids}")
    print(f"  first_eos_index={first_eos}")
    print(f"  reward_text={reward_text!r}")
    print(f"  eos_then_continues={metadata['eos_then_continues']}")
    print(f"  eos_first={metadata['eos_first']}")
    print(f"  valid_response_length={metadata['valid_response_length']}")
    print(f"  training_labels={labels}")
    print(f"  training_p_mask={p_mask}")
    return metadata, reward_text, labels, p_mask


def main():
    metadata, reward_text, labels, p_mask = run_case(
        "Case A", [A, B, EOS, C, D], "a b <EOS> c d"
    )
    assert reward_text == "a b "
    assert labels == [A, B, EOS, EOS, EOS]
    assert p_mask == [True, True, True, True, True]
    assert metadata["eos_then_continues"] is True

    metadata, reward_text, labels, p_mask = run_case(
        "Case B", [EOS, A, B], "<EOS> a b"
    )
    assert reward_text == ""
    assert metadata["eos_first"] is True
    assert labels == [EOS, EOS, EOS]
    assert p_mask == [True, True, True]

    metadata, reward_text, labels, p_mask = run_case(
        "Case C", [A, B, C], "a b c"
    )
    assert reward_text == "a b c"
    assert metadata["missing_eos"] is True
    assert labels == [A, B, C]
    assert p_mask == [True, True, True]

    print("EOS smoke checks passed.")


if __name__ == "__main__":
    main()

