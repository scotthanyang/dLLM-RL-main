from rl_eos import eos_response_metadata, pad_after_first_eos, truncate_text_at_first_eos


PAD = 0
EOS = 2
A = 10
B = 11
C = 12
D = 13


def training_mask(response_ids, supervise_first_eos=True):
    labels = list(response_ids)
    p_mask = [True] * len(response_ids)
    try:
        eos_idx = response_ids.index(EOS)
    except ValueError:
        return labels, p_mask

    inactive_start = eos_idx + 1 if supervise_first_eos else eos_idx
    for idx in range(inactive_start, len(response_ids)):
        labels[idx] = -100
        p_mask[idx] = False
    return labels, p_mask


def run_case(name, response_ids, response_text):
    padded_ids, first_eos = pad_after_first_eos(response_ids, [EOS], PAD)
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
    labels, p_mask = training_mask(response_ids, supervise_first_eos=True)
    print(f"{name}:")
    print(f"  raw_ids={response_ids}")
    print(f"  padded_after_eos={padded_ids}")
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
    assert labels == [A, B, EOS, -100, -100]
    assert p_mask == [True, True, True, False, False]
    assert metadata["eos_then_continues"] is True

    metadata, reward_text, labels, p_mask = run_case(
        "Case B", [EOS, A, B], "<EOS> a b"
    )
    assert reward_text == ""
    assert metadata["eos_first"] is True
    assert labels == [EOS, -100, -100]
    assert p_mask == [True, False, False]

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

