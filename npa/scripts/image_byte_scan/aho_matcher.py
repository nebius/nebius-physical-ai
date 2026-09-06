"""Exact finite byte policy with one Aho-Corasick scan per buffer.

No serialization/pickle, implicit dependency installation, or fallback matcher.
Dependency is loaded by the controlled runner from the verified sealed descriptor.
"""
from dataclasses import dataclass, field
from hashlib import sha256

import ahocorasick

POLICY = "exact-or-short-ascii-token-v1"
EXACT = "exact-substring-v1"
ASCII_WORD = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


@dataclass(frozen=True, slots=True)
class CompiledLiterals:
    carry: int
    count: int
    policy: str
    _automaton: object = field(repr=False, compare=False)

    def occurrences(self, raw):
        # Latin-1 is a bijection byte -> code point: no UTF-8 decoding/replacement,
        # normalization, loss of NUL, or invalid-sequence omission is possible.
        if self.count:
            yield from self._automaton.iter(raw.decode("latin-1"))


def compile_literals(values, policy):
    if policy not in {POLICY, EXACT}:
        raise ValueError("literal_matching_policy")
    if not isinstance(values, (list, tuple)) or any(type(v) is not str or not v for v in values):
        raise ValueError("literal_inventory_schema")
    if ahocorasick.unicode != 1:
        raise ValueError("unexpected_extension_unicode_mode")
    groups = {}
    carry = 1
    for index, value in enumerate(values):
        raw = value.encode("utf-8")
        carry = max(carry, len(raw) + 1)
        # Preserve duplicate input patterns as distinct indexed policy entries.
        groups.setdefault(raw, []).append((index, sha256(raw).hexdigest(), len(raw),
                                          policy == POLICY and len(value) < 6))
    automaton = ahocorasick.Automaton()
    for raw, entries in groups.items():
        automaton.add_word(raw.decode("latin-1"), tuple(entries))
    if groups:
        automaton.make_automaton()
    return CompiledLiterals(carry, len(values), policy, automaton)


class LiteralMatcher:
    """Per-record mutable cursor, sharing only a compiled read-only policy."""
    def __init__(self, compiled):
        if type(compiled) is not CompiledLiterals:
            raise TypeError("compiled_literal_policy_required")
        self.compiled = compiled
        self.buffer, self.base = b"", 0
        self.next_positions = [0] * compiled.count

    def feed(self, data, *, final=False):
        if type(data) is not bytes or type(final) is not bool:
            raise TypeError("literal_feed_schema")
        self.buffer += data
        boundary = len(self.buffer) if final else max(0, len(self.buffer) - self.compiled.carry)
        found = []
        for end_index, entries in self.compiled.occurrences(self.buffer):
            local_end = end_index + 1
            for index, digest, length, bounded in entries:
                local_start = local_end - length
                start, end = self.base + local_start, self.base + local_end
                if local_start >= boundary or start < self.next_positions[index]:
                    continue
                if bounded:
                    if local_start and self.buffer[local_start - 1] in ASCII_WORD:
                        continue
                    if local_end < len(self.buffer) and self.buffer[local_end] in ASCII_WORD:
                        continue
                found.append({"rule_id": "private_literal", "literal_index": index,
                              "literal_sha256": digest, "byte_start": start, "byte_end": end})
                # re.finditer suppresses self-overlaps for each individual pattern;
                # other patterns and duplicate entries retain independent cursors.
                self.next_positions[index] = end
        floor = self.base + boundary
        for index, position in enumerate(self.next_positions):
            self.next_positions[index] = max(position, floor)
        trim = max(0, boundary - 1)
        self.buffer = self.buffer[trim:]
        self.base += trim
        # Baseline emits in policy-index then left-to-right order for EACH feed.
        found.sort(key=lambda row: (row["literal_index"], row["byte_start"]))
        return found
