def _entry(x0: float, x1: float, probability: float) -> dict:
    return {'x0': round(x0, 4), 'x1': round(x1, 4), 'c': int(round(probability * 100))}


def character_profile(char_token_counts, token_spans, total_frames):
    """One confidence band per character, positioned where its ink actually is.

    token_spans holds (start_frame, end_frame, probability) per target token, in order,
    from the CTC forced alignment - torchaudio's spans, whose end frame is exclusive. char_token_counts says how many tokens each character
    consumed, which is above one for a byte-level vocabulary. x is a fraction of the line
    width, so the client can place the band without knowing the crop's pixel size.

    A character scores the minimum of its tokens: it is only as right as its worst byte.
    Returns None when the inputs disagree, so the caller can fall back to uniform spacing.
    """
    if not char_token_counts and not token_spans:
        return []
    if total_frames <= 0 or sum(char_token_counts) != len(token_spans):
        return None

    profile = []
    position = 0
    for count in char_token_counts:
        group = token_spans[position:position + count]
        position += count
        start = min(span[0] for span in group)
        end = max(span[1] for span in group)
        profile.append(_entry(start / total_frames,
                              min(1.0, end / total_frames),
                              min(span[2] for span in group)))
    return profile


def uniform_profile(probabilities):
    """The legacy spacing: every entry gets an equal share of the line width. Used when
    alignment is unavailable, and kept so old rows keep rendering the way they always did."""
    count = len(probabilities)
    return [_entry(i / count, (i + 1) / count, p) for i, p in enumerate(probabilities)]
