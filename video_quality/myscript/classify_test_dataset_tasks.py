"""Classify WorldArena test_dataset episodes into RoboTwin 2.0 Clean-50 tasks.

The test_dataset ships 1000 episodes under ``fixed_scene_task/`` whose hdf5
files carry only action tensors (no task label) and whose json instructions
only contain free-form English prompts. RoboTwin 2.0 Clean-50 has 50 tasks
with 20 test episodes per task (= 1000 total), so this script maps each
prompt to one of the 50 tasks via hand-crafted keyword rules and then
writes three artifacts:

    - ``task_counts.md``          per-task episode counts (two sort orders)
    - ``episode_task_mapping.csv`` full 1000-row mapping
    - ``unclassified.csv``        any episode that fell through all rules

The mapping is heuristic and may miss the occasional episode that phrasing
the prompt with unusual wording; use ``unclassified.csv`` and the count
table to spot-check and iterate.

Usage:
    python video_quality/myscript/classify_test_dataset_tasks.py \\
        [--dataset-root /mydir/code/WorldArena/datasets/WorldArena_Robotwin2.0/test_dataset] \\
        [--output-dir video_quality/myscript/task_classification_output]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROBOTWIN_CLEAN50_TASKS: list[str] = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]

_COMMON_PREFIX = (
    "in a fixed robotic workspace, generate a rigid, physically consistent "
    "embodied robotic arm. the arm maintains high stability with no "
    "deformation and enters the frame to"
)

_HORIZONTAL_SHAKE_MARKERS = (
    "horizontal",
    "horizontally",
    "sideways",
    "side to side",
    "side-to-side",
    "back and forth",
)

_STRICT_DUAL_BOTTLE_MARKERS = (
    "both arms",
    "with both arms",
    "each arm",
    "with each arm",
    "use each arm",
    "let each arm",
    "dual arm",
    "separate arms",
    "separate hands",
    "simultaneously",
    "in one arm, the",
    "in one hand, the",
    "in one arm and",
    "in one hand and",
    "the other arm",
    "the opposite hand",
    "the opposite arm",
    "with both hands",
    "catch both ",
    "both the ",
    "together with",
    " too.",
    " too ",
)

_HANDOVER_BLOCK_MARKERS = (
    "hand it to the right",
    "hand it over",
    "hand it to the other",
    "handover",
    "hand over",
    "give it to the right",
    "give it to the other",
    "pass it to the right",
    "pass it into",
    "pass it across",
    "pass it over",
    "pass it to the other",
    "pass it to the",
    "to the right arm, and place",
    "to the other arm, and place",
    "switch it from",
    "transfer it into the right",
    "transfer it to the right",
    "transfer it to the other",
    "transfer to the right arm",
    "transfer to the other arm",
    "transfer it into",
    "transfer it,",
    "transfer it and",
    "blue pad",
    "move it to the right, and place",
)

# Manual overrides for episodes whose text-only instruction is genuinely
# ambiguous (often because the prompt is truncated before its destination
# noun) but whose task can be confirmed from the first-frame image under
# ``test_dataset/first_frame/fixed_scene_task/``.
_MANUAL_EPISODE_OVERRIDES: dict[int, str] = {
    # "...and position it" — first frame shows the small pad next to the
    # pill bottle, so this is move_pillbottle_pad, not adjust_bottle.
    304: "move_pillbottle_pad",
    # "...and move it" — first frame also shows the pad next to the
    # storage bottle; move_pillbottle_pad, not adjust_bottle.
    458: "move_pillbottle_pad",
    # "Lift the bottle the dark drink bottle with logo up from the table"
    # is a single-bottle prompt with a repeated appositive; bottle_count
    # heuristic misreads it as two bottles. First frame shows only one
    # bottle, so this belongs to adjust_bottle.
    582: "adjust_bottle",
}

_HAMMER_BEAT_MARKERS = (
    "beat",
    "strike",
    "hit",
    "hammer the block",
    "pound block",
    "pound the block",
    "use it on the block",
    "use on the block",
    "on the block",
    "block with it",
)


def _clean_instruction(text: str) -> str:
    text = text.strip().lower()
    if text.startswith(_COMMON_PREFIX):
        text = text[len(_COMMON_PREFIX):].strip()
    return text


def _has(text: str, *needles: str) -> bool:
    return any(n in text for n in needles)


def _has_left_of(text: str) -> bool:
    # Phrases that clearly describe "place A relative to B on the left" without
    # accidentally matching unrelated tokens like "left arm", "left to right".
    return _has(
        text,
        "to the left of",
        "left of the ",
        "left side of the",
        "left position of",
        "on the left side of",
        "left side of",
        "leftward to the side",
        "'s left side",
        "'s left position",
        "'s left.",
        "'s left ",
        "'s left",
        "left of ",
        "leftmost side",
    )


def _has_right_of(text: str) -> bool:
    return _has(
        text,
        "to the right of",
        "right of the ",
        "right side of the",
        "right position of",
        "on the right side of",
        "right side of",
        "'s right side",
        "'s right position",
        "'s right.",
        "'s right ",
        "'s right",
        "right of ",
        "rightmost side",
        # note: we deliberately exclude "right next to" here so that
        # instructions like "set the can right next to the pot" flow into
        # the move_can_pot rule instead of being hijacked by place_a2b.
    )


def _count_color_blocks(text: str) -> tuple[int, bool, bool, bool]:
    red = "red block" in text
    green = "green block" in text
    blue = "blue block" in text
    return int(red) + int(green) + int(blue), red, green, blue


def _count_size_blocks(text: str) -> int:
    return sum(
        c in text for c in ("large block", "medium block", "small block")
    )


def classify(raw_instruction: str) -> str:
    """Return a RoboTwin 2.0 Clean-50 task name for ``raw_instruction``.

    Returns ``"UNCLASSIFIED"`` if no rule matched so the caller can surface
    the episode for manual review. The priority order matters: we resolve
    tasks whose defining action is unambiguous (handover of a microphone,
    stamping a seal, pressing a clock) first, then target-location
    placements (X onto electronicscale / displaystand / coaster / ...),
    then generic A-relative-to-B placements, and finally fallback buckets
    for bottles/blocks/bowls.
    """

    t = _clean_instruction(raw_instruction)

    # ---- 0. Relative-placement short-circuit (A left/right of B) ----
    # Must come before single-object rules so "bell left of phone" goes to
    # place_a2b_left, not click_bell.  We skip this short-circuit only when
    # the instruction is clearly a multi-color block arrangement task
    # (blocks_ranking_rgb or blocks_ranking_size), which also uses the word
    # "left/right" in phrases like "left to right".
    color_block_count, _, _, _ = _count_color_blocks(t)
    size_block_count = _count_size_blocks(t)
    is_block_arrangement = color_block_count >= 2 or size_block_count >= 2

    left_of = _has_left_of(t)
    right_of = _has_right_of(t)
    if (left_of or right_of) and not is_block_arrangement:
        if right_of and not left_of:
            return "place_a2b_right"
        if left_of and not right_of:
            return "place_a2b_left"
        # Both present: defer to right-of for the "A on right, B on left" style.
        return "place_a2b_right"

    # ---- 1. Strong single-object action tasks ----

    if "roller" in t:
        return "grab_roller"

    if "microphone" in t:
        return "handover_mic"

    if "laptop" in t:
        return "open_laptop"

    if "microwave" in t or "appliance with transparent glass" in t:
        return "open_microwave"

    if "scanner" in t or "scans the" in t:
        return "scan_object"

    if _has(t, "paymentsign", "payment display", "payment stand",
            "stand for payments", "payment logo"):
        return "rotate_qrcode"

    if _has(t, "qrcode", "qr code") and _has(t, "rotate", "align", "face",
                                              "adjust", "turning"):
        return "rotate_qrcode"

    if "mug" in t and _has(t, "rack", "hang"):
        return "hanging_mug"

    # Hammer+block: broad verbs for beat/strike/pound/hammer.
    if "hammer" in t and (
        _has(t, *_HAMMER_BEAT_MARKERS)
        or ("block" in t and _has(t, "use it", "use on", "use the hammer",
                                    "then pound", "then hammer", "to hammer",
                                    "then beat", "then strike", "then hit"))
    ):
        return "beat_block_hammer"

    # Stamp seal: seal/stamping-tool combined with a stamping verb that
    # targets a color word (e.g. "stamp Beige", "mark Tan", "press on
    # Orange", "apply to Blue", "seal for Red", "onto Brown").
    if "seal" in t or "stamping tool" in t:
        _stamp_verbs = (
            "stamp ", "stamps ", "stamp.", "stamping",
            "mark ", "press on ", "press down on ", "apply to ",
            "apply on ", "apply it onto ", "apply onto ",
            "align it with ", "align with ", "seal for ", "onto ",
            "on tan", "on beige", "on silver", "on red", "on blue",
            "on black", "on brown", "on orange", "on gray", "on maroon",
            "on yellow", "on magenta", "on pink", "on green", "on purple",
            "on cyan", "on white", "on gold",
            "press beige", "press tan", "press silver", "press red",
            "press blue", "press black", "press brown", "press orange",
            "press gray", "press maroon", "press yellow", "press magenta",
            "press pink", "press green", "press purple", "press cyan",
            "press white", "press gold", "press coral",
        )
        if _has(t, *_stamp_verbs):
            return "stamp_seal"

    # Alarm-clock-specific phrasing (kept before bell so "clock" beats bell ambiguity).
    if _has(t, "alarm-clock", "alarmclock", "alarm clock", "clock with "):
        return "click_alarmclock"

    if _has(t, "playingcard", "playing card", "cards box",
            "cards inside", "cards pack", "cards container",
            "cards carton", "cards packaging"):
        # Refine: if the cards go into a basket or a cabinet, or next to
        # something, classify accordingly.
        if _has(t, "cabinet", "drawer") and _has(t, "into"):
            return "put_object_cabinet"
        if _has(t, "basket", "handheld basket"):
            return "place_object_basket"
        return "move_playingcard_away"

    if _has(t, "trashbin", "bin for waste") and _has(
        t, "pour", "empty", "tilt", "contents", "balls out", "balls inside",
        "transfer balls", "pour into", "pour its", "pour all"
    ):
        return "dump_bin_bigbin"

    if "bin with flared" in t and "pour" in t:
        return "dump_bin_bigbin"

    # ---- 2. Target-location generic "place object on destination" rules ----

    if "electronicscale" in t or "electronic scale" in t:
        return "place_object_scale"

    # Skip stand rule when the primary action clearly targets a bell
    # (e.g. "touch the bell with black stand's top" is click_bell).
    _bell_click_with_stand = "bell" in t and _has(
        t, "touch", "tap", " click", "click ", "press"
    ) and ("stand's top" in t or "stand 's top" in t)
    if not _bell_click_with_stand and _has(
        t, "displaystand", "display stand", "black smooth stand",
        "smooth stand", "matte black stand", "plastic black stand",
        "metal stand", "wedge-shaped stand", "hollow black stand",
        "angled black smooth stand", "rectangular plastic stand",
        "simple black stand",
    ):
        return "place_object_stand"

    if _has(t, "phonestand", "phone holder", "phone stand", "phone rack",
            "holder with clamp", "plastic holder"):
        return "place_phone_stand"

    if "coaster" in t:
        return "place_empty_cup"

    if _has(t, "shoe-box", "shoebox", "shoe box", "box for shoes",
            "nike swoosh", "box with nike", "orange box", "orange shoe",
            "medium-sized box"):
        if _has(t, "shoe", "footwear", "sneaker", "slipper"):
            return "place_dual_shoes"

    if "breadbasket" in t:
        return "place_bread_basket"

    if ("can" in t or "cans" in t) and _has(
        t, "plasticbox", "plastic box", "box for storage", "blue box",
        "deep blue box", "same box", "box with angled sides",
    ):
        return "place_cans_plasticbox"

    if ("can" in t or "cans" in t) and _has(
        t, "kitchenpot", "kitchen pot", "silver pot", "cooking pot",
        "round pot", "gray pot", "black pot", "teal blue cooking",
        "dark gray knob", "near the pot", "next to the pot",
        "beside the", "next to the silver", "next to the black",
        "next to the gray", "next to the medium", "next to the teal",
        "near the silver", "near the black", "near the gray",
        "near the medium", "near the teal",
    ) and "pot" in t:
        return "move_can_pot"

    if ("can" in t or "cans" in t) and "basket" in t:
        return "place_can_basket"

    if _has(t, "hamburg", "burger"):
        return "place_burger_fries"

    if _has(t, "bread", "loaf", "baguette"):
        if _has(t, "skillet", "cooking pan", "frying pan") or " pan" in t:
            return "place_bread_skillet"
        if _has(t, "cabinet", "drawer"):
            return "put_object_cabinet"
        if "basket" in t:
            return "place_bread_basket"
        return "place_bread_basket"

    if _has(t, "shoe", "footwear", "sneaker", "slipper"):
        if _has(t, "two ", "both the ", "two the ", "pair", "into the medium",
                "into the orange", "into the nike"):
            return "place_dual_shoes"
        return "place_shoe"

    if _has(t, "cabinet", "drawer"):
        return "put_object_cabinet"

    # ---- 3. Bottle-centric tasks ----

    if "bottle" in t:
        if _has(t, "dustbin", "trash bin", "trash holder", "trash container",
                "garbage bin", "garbage container", "garbage can", "trash can"):
            return "put_bottles_dustbin"
        if ("shake" in t or "shaking" in t
                or "move it back and forth" in t
                or "side-to-side" in t
                or "move it horizontally" in t
                or "move horizontally" in t):
            if _has(t, *_HORIZONTAL_SHAKE_MARKERS):
                return "shake_bottle_horizontally"
            return "shake_bottle"
        if _has(t, "head-up", "head up", "upright", "stand upright"):
            return "adjust_bottle"
        if "pad" in t and "bottle" in t:
            return "move_pillbottle_pad"
        bottle_count = t.count("bottle")
        if bottle_count >= 2 and _has(t, *_STRICT_DUAL_BOTTLE_MARKERS):
            return "pick_dual_bottles"
        if bottle_count >= 2:
            return "pick_diverse_bottles"
        # Single-bottle fall-through: instructions describing picking / lifting
        # one bottle without an explicit re-orient verb still belong to
        # adjust_bottle in Clean-50 (it's the only single-bottle task that
        # isn't shake / dustbin / pad).
        return "adjust_bottle"

    # ---- 4. Single-object fallback tasks ----

    # Rare phrasing: "gray base with black buttons" describes the switch
    # console without using the literal word "switch".
    if _has(t, "gray base with black buttons", "base with black buttons"):
        return "turn_switch"

    if "switch" in t:
        # Disambiguate from handover_block: "switch arms", "switch it from
        # the left arm", etc. are bimanual block transfers that happen to
        # mention the word switch.
        if "block" in t and _has(
            t, "switch arms", "switch arm", "switch it from",
            "switch it to", "switch to the"
        ):
            return "handover_block"
        return "turn_switch"

    if "bell" in t:
        return "click_bell"

    if "fan" in t:
        return "place_fan"

    if "mug" in t:
        return "hanging_mug"

    # ---- 5. Container + plate (bowl is the vehicle) ----

    if "container" in t and ("plate" in t or "bowl" in t):
        return "place_container_plate"
    if "container" in t and "pad" in t:
        return "move_pillbottle_pad"
    if "container" in t and _has(t, "basket"):
        return "place_object_basket"
    if "container" in t and _has(t, "trashbin", "pour", "bin"):
        return "dump_bin_bigbin"
    if "container" in t:
        return "place_container_plate"

    # ---- 6. Pot lift ----

    if _has(t, "kitchenpot", "metal pot", "cooking pot", " pot ",
            "pot upright") and _has(t, "lift", "raise", "bring ", "bringing"):
        return "lift_pot"

    # ---- 7. Bowls stacking ----

    if "bowl" in t:
        three_bowl_markers = (
            "three",
            "three the ",
            "biggest to smallest",
            "smallest to biggest",
            "sequentially",
            "previous one",
            "the two above",
            "the remaining two",
            "next two above",
            "next two",
            "one on top of another",
            "on top of each other",
            "all three",
            "one after another",
            "the second",
            "the third",
            "stack the rest",
            "position the second",
            "position the third",
            "rest on it",
        )
        if _has(t, *three_bowl_markers):
            return "stack_bowls_three"
        return "stack_bowls_two"

    # ---- 8. Block tasks ----

    color_count, red_b, green_b, blue_b = _count_color_blocks(t)
    size_count = _count_size_blocks(t)

    if "block" in t and _has(t, *_HANDOVER_BLOCK_MARKERS):
        return "handover_block"

    stack_like = _has(
        t,
        "stack",
        "pile ",
        "arrange blue block on",
        "arrange green block on",
        "on top of green block",
        "on top of red block",
        "over green block",
        "over red block",
        "blue block on green block",
        "green block on red block",
    )

    if color_count == 3:
        if stack_like:
            return "stack_blocks_three"
        return "blocks_ranking_rgb"

    if color_count == 2:
        if stack_like or "on top of" in t:
            return "stack_blocks_two"
        return "stack_blocks_two"

    if size_count == 3:
        if stack_like:
            return "stack_blocks_three"
        return "blocks_ranking_size"

    if color_count == 1 and "block" in t:
        # Single-color block with no multi-block stacking context is the
        # defining pattern of handover_block (red block, transferred between
        # arms). Even terse phrasings like "Move the red block using the
        # left arm" belong here, because no other Clean-50 task deals with
        # a single red block alone.
        return "handover_block"

    # ---- 9. Mouse / toy car / generic basket placements ----

    if "mouse" in t:
        return "place_mouse_pad"

    if _has(t, "toycar", "toy car"):
        if "basket" in t:
            return "place_object_basket"
        return "place_object_basket"

    if "basket" in t:
        return "place_object_basket"

    # ---- 10. Stapler ----

    if "stapler" in t:
        if _has(t, " mat", "colored mat", "on gray mat", "on blue mat",
                "on red mat", "on black mat", "on magenta mat",
                "on yellow mat", "on cyan mat", "on green mat",
                "on the mat", "to the magenta mat", "to the gray mat",
                "to the red mat", "to the blue mat", "to the black mat",
                "to the cyan mat", "to the yellow mat",
                "to the magenta colored mat", "to the gray colored mat",
                "shift it to "):
            return "move_stapler_pad"
        if _has(t, "press", "apply pressure", "apply force", "engage",
                "operate", "push down", "push the", "push top", "staple",
                "activate it", "firmly press", "simply press",
                "push down on", "push ", "lower the", " lower "):
            return "press_stapler"
        return "press_stapler"

    # ---- 11. Block / pad fallback ----

    if "block" in t and "pad" in t:
        return "handover_block"
    if "block" in t:
        return "stack_blocks_two"

    return "UNCLASSIFIED"


def load_instructions(instructions_dir: Path) -> dict[int, str]:
    data: dict[int, str] = {}
    for path in instructions_dir.glob("episode*.json"):
        stem = path.stem
        try:
            episode_id = int(stem[len("episode"):])
        except ValueError:
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        text = (
            payload.get("instruction")
            or payload.get("instruction_1")
            or payload.get("instruction_2")
            or ""
        )
        data[episode_id] = text
    return data


def write_outputs(
    output_dir: Path,
    classified: list[tuple[int, str, str]],
    counts: Counter,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_csv = output_dir / "episode_task_mapping.csv"
    with mapping_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "predicted_task", "instruction"])
        for episode_id, task, instruction in sorted(classified, key=lambda x: x[0]):
            writer.writerow([f"episode{episode_id}", task, instruction])

    unclassified_csv = output_dir / "unclassified.csv"
    with unclassified_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "instruction"])
        for episode_id, task, instruction in sorted(classified, key=lambda x: x[0]):
            if task == "UNCLASSIFIED":
                writer.writerow([f"episode{episode_id}", instruction])

    counts_md = output_dir / "task_counts.md"
    lines: list[str] = []
    total = sum(counts.values())
    predicted_tasks = [k for k in counts if k != "UNCLASSIFIED"]
    lines.append("# Test Dataset Task Distribution")
    lines.append("")
    lines.append(f"- Total episodes: **{total}**")
    lines.append(f"- Unique tasks predicted: **{len(predicted_tasks)} / 50**")
    if counts.get("UNCLASSIFIED", 0):
        lines.append(f"- Unclassified episodes: **{counts['UNCLASSIFIED']}**")
    missing = [task for task in ROBOTWIN_CLEAN50_TASKS if task not in counts]
    if missing:
        lines.append(f"- Clean-50 tasks with zero predictions: `{missing}`")
    lines.append("")
    lines.append("## Sorted by count (desc)")
    lines.append("")
    lines.append("| Task | Count |")
    lines.append("|------|-------|")
    for task, count in counts.most_common():
        lines.append(f"| {task} | {count} |")
    lines.append("")
    lines.append("## Sorted by task name")
    lines.append("")
    lines.append("| Task | Count |")
    lines.append("|------|-------|")
    for task in sorted(counts.keys()):
        lines.append(f"| {task} | {counts[task]} |")
    counts_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/mydir/code/WorldArena/datasets/WorldArena_Robotwin2.0/test_dataset"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "task_classification_output",
    )
    parser.add_argument(
        "--instruction-variant",
        default="instructions",
        help="instructions | instructions_1 | instructions_2",
    )
    args = parser.parse_args()

    instructions_dir: Path = (
        args.dataset_root / args.instruction_variant / "fixed_scene_task"
    )
    if not instructions_dir.is_dir():
        raise SystemExit(f"Instructions dir not found: {instructions_dir}")

    instructions = load_instructions(instructions_dir)
    if not instructions:
        raise SystemExit(f"No episode*.json found under {instructions_dir}")

    classified: list[tuple[int, str, str]] = []
    counts: Counter = Counter()
    for episode_id, raw in instructions.items():
        task = _MANUAL_EPISODE_OVERRIDES.get(episode_id) or classify(raw)
        counts[task] += 1
        classified.append((episode_id, task, raw))

    write_outputs(args.output_dir, classified, counts)

    total = sum(counts.values())
    unclassified = counts.get("UNCLASSIFIED", 0)
    predicted_tasks = [k for k in counts if k != "UNCLASSIFIED"]

    print(f"Episodes processed: {total}")
    print(f"Unique tasks predicted: {len(predicted_tasks)} / 50")
    print(f"Unclassified: {unclassified}")
    print("")
    print(f"{'Count':>6}  Task")
    print(f"{'-----':>6}  ----")
    for task, count in counts.most_common():
        marker = ""
        if task != "UNCLASSIFIED" and (count < 15 or count > 30):
            marker = "  <== deviates from ~20"
        print(f"{count:>6}  {task}{marker}")

    missing = [t for t in ROBOTWIN_CLEAN50_TASKS if t not in counts]
    if missing:
        print("")
        print(f"Clean-50 tasks with zero predictions: {missing}")

    print("")
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
