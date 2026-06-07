"""
Reusable MIDI motif library + converters for the immersive-audio pipeline.

This module is the single source of truth for the per-mood melodies. It exposes:

  MELODY_LIBRARY            mood -> motif, where a motif is a list of
                            (pitch, hold_frames, gap_frames) tuples.
  build_segments(motif)     motif -> MRT2 render() note segments (128-int
                            conditioning vectors) for Modal/Magenta.
  melody_for_mood(mood)     convenience: segments for a mood (neutral fallback).
  motif_to_note_events()    motif -> [(pitch, start_s, duration_s)] — a plain,
                            framework-agnostic note list.
  export_midi(motif, path)  write a standard .mid file (pure Python, no deps)
                            so a motif can be saved, inspected, or reused.

Timing: 1 frame = 40 ms @ 25 Hz (the MRT2 convention).
MIDI pitches: C4=60, D4=62, Eb4=63, E4=64, F4=65, G4=67, Ab4=68, A4=69,
Bb4=70, B4=71, C5=72; lower octave C3=48, G3=55.
"""

from __future__ import annotations

import struct
from pathlib import Path

FRAME_MS = 40  # one MRT2 frame = 40 ms
NOTE_VELOCITY = 90

# ── Curated mood -> MIDI motif library ───────────────────────────────────────
# Each motif: list of (pitch, hold_frames, gap_frames).
MELODY_LIBRARY: dict[str, list] = {
    # gentle rising major arpeggio, long holds
    "calm": [(60, 20, 6), (64, 20, 6), (67, 26, 8), (72, 30, 12)],
    # low, repeated, chromatic, short and clipped
    "tense": [(48, 6, 2), (49, 6, 2), (48, 6, 2), (51, 10, 4), (48, 12, 6)],
    # fast driving riff, minimal gaps
    "action": [(60, 5, 1), (67, 5, 1), (72, 5, 1), (67, 5, 1),
               (60, 5, 1), (63, 5, 1), (67, 8, 2)],
    # slow minor descending line (A minor)
    "sad": [(69, 26, 8), (67, 26, 8), (65, 26, 8), (64, 32, 12)],
    # sparse whole-tone-ish, airy spacing
    "mysterious": [(62, 14, 10), (66, 14, 10), (70, 18, 12), (64, 22, 14)],
    # ascending major fanfare
    "triumphant": [(60, 8, 2), (64, 8, 2), (67, 8, 2), (72, 14, 4), (76, 20, 6)],
}

# ── Built-in MARY ("Mary Had a Little Lamb") motif — neutral fallback ────────
_E4, _D4, _C4, _G4 = 64, 62, 60, 67
_BEEP = 84  # C6 — sharp marker note between phrases


def _phrase(notes):
    """Append a BEEP marker after a phrase."""
    return list(notes) + [(_BEEP, 3, 8)]


_MARY = [
    *_phrase([(_E4, 13, 2), (_D4, 13, 2), (_C4, 13, 2), (_D4, 13, 2),
              (_E4, 13, 2), (_E4, 13, 2), (_E4, 26, 4)]),
    *_phrase([(_D4, 13, 2), (_D4, 13, 2), (_D4, 26, 4),
              (_E4, 13, 2), (_G4, 13, 2), (_G4, 26, 4)]),
    *_phrase([(_E4, 13, 2), (_D4, 13, 2), (_C4, 13, 2), (_D4, 13, 2),
              (_E4, 13, 2), (_E4, 13, 2), (_E4, 13, 2), (_E4, 26, 4)]),
    *_phrase([(_D4, 13, 2), (_D4, 13, 2), (_E4, 13, 2), (_D4, 13, 2), (_C4, 26, 4)]),
    (_D4, 13, 2), (_D4, 13, 2), (_E4, 13, 2), (_D4, 13, 2), (_C4, 26, 8),
]

MELODY_LIBRARY["neutral"] = _MARY


def build_segments(motif: list) -> list:
    """Convert a (pitch, hold, gap) motif into MRT2 render() note segments.

    Each segment: {"notes": [128 ints], "drums": [int], "frames": int},
    matching MagentaInference.render() in modal_magenta.py. 2 = note onset,
    1 = held note, 0 = silent.
    """
    segments = []
    for pitch, hold, gap in motif:
        onset = [0] * 128
        onset[pitch] = 2
        segments.append({"notes": onset, "drums": [-1], "frames": 1})

        if hold > 1:
            cont = [0] * 128
            cont[pitch] = 1
            segments.append({"notes": cont, "drums": [-1], "frames": hold - 1})

        if gap > 0:
            segments.append({"notes": [0] * 128, "drums": [-1], "frames": gap})

    return segments


def melody_for_mood(mood: str) -> list:
    """Return the render() note segments for a mood, falling back to neutral."""
    return build_segments(MELODY_LIBRARY.get(mood, MELODY_LIBRARY["neutral"]))


# ── Key transposition helpers ─────────────────────────────────────────────────

KEY_SEMITONES: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def transpose_segments(segments: list, semitones: int) -> list:
    """Shift all note onsets/holds in a segment list by `semitones`."""
    if semitones == 0:
        return segments
    out = []
    for seg in segments:
        new_notes = [0] * 128
        for i, v in enumerate(seg["notes"]):
            if v > 0:
                j = i + semitones
                if 0 <= j < 128:
                    new_notes[j] = v
        out.append({**seg, "notes": new_notes})
    return out


def motif_to_note_events(motif: list, frame_ms: int = FRAME_MS) -> list:
    """Convert a motif into a plain [(pitch, start_s, duration_s)] note list.

    Framework-agnostic — useful for previewing, scheduling, or feeding another
    synth/DAW. Gaps advance the clock without emitting a note.
    """
    events = []
    clock_ms = 0
    for pitch, hold, gap in motif:
        start_s = clock_ms / 1000.0
        duration_s = (hold * frame_ms) / 1000.0
        events.append((pitch, start_s, duration_s))
        clock_ms += (hold + gap) * frame_ms
    return events


# ── Standard MIDI File export (pure Python, no external deps) ─────────────────
# Type-0 SMF with division = 25 ticks/quarter and tempo = 1,000,000 us/quarter
# (60 BPM). One quarter note = 1 s = 25 ticks, so 1 tick = 40 ms = exactly one
# MRT2 frame: frame counts map 1:1 to MIDI ticks with no rounding.
_TICKS_PER_QUARTER = 25
_TEMPO_US_PER_QUARTER = 1_000_000


def _vlq(value: int) -> bytes:
    """Encode an int as a MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("VLQ cannot encode negative values")
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(out)


def export_midi(motif: list, path, velocity: int = NOTE_VELOCITY) -> str:
    """Write `motif` to a standard .mid file. Returns the path written.

    The result is a real Standard MIDI File any DAW or MIDI tool can open,
    so motifs from MELODY_LIBRARY can be saved, inspected, or reused.
    """
    track = bytearray()

    # Tempo meta event (delta 0): FF 51 03 tttttt
    track += _vlq(0) + b"\xff\x51\x03" + struct.pack(">I", _TEMPO_US_PER_QUARTER)[1:]

    pending_rest = 0  # ticks of silence to absorb into the next event's delta
    for pitch, hold, gap in motif:
        track += _vlq(pending_rest) + bytes([0x90, pitch & 0x7F, velocity & 0x7F])  # note on
        track += _vlq(hold) + bytes([0x80, pitch & 0x7F, 0])                        # note off
        pending_rest = gap

    # End of track meta event
    track += _vlq(pending_rest) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, _TICKS_PER_QUARTER)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)

    path = Path(path)
    path.write_bytes(header + chunk)
    return str(path)


def export_all_moods(out_dir, velocity: int = NOTE_VELOCITY) -> list:
    """Write one .mid file per mood in MELODY_LIBRARY. Returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [export_midi(motif, out_dir / f"{mood}.mid", velocity)
            for mood, motif in MELODY_LIBRARY.items()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export mood motifs to .mid files")
    parser.add_argument("--out", default="midi_out", help="Output directory")
    parser.add_argument("--mood", default=None,
                        help=f"Export a single mood ({', '.join(MELODY_LIBRARY)}); default: all")
    args = parser.parse_args()

    if args.mood:
        if args.mood not in MELODY_LIBRARY:
            parser.error(f"unknown mood {args.mood!r}; choose from {list(MELODY_LIBRARY)}")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        print("Wrote", export_midi(MELODY_LIBRARY[args.mood], Path(args.out) / f"{args.mood}.mid"))
    else:
        for p in export_all_moods(args.out):
            print("Wrote", p)
