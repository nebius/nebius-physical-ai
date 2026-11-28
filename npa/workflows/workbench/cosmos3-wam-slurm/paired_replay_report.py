"""Compare native LIBERO-10 episodes and diagnose paired failures from ordered CPU replays."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

DRAWER = ("close", "white_cabinet_1_bottom_region")
BOWL = ("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region")
PUDDING = ("on", "chocolate_pudding_1", "living_room_table_plate_right_region")
MUG = ("on", "porcelain_mug_1", "plate_1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def add_native_episodes(path, task, group, rows, trials):
    episodes = group["episode_results"]
    for e in episodes:
        state, success, steps = e["episode"], e["success"], e["steps"]
        key = (task, state)
        require(
            integer(state) and state < trials and key not in rows,
            f"{path}: invalid or duplicate episode {key}",
        )
        require(
            type(success) is bool
            and integer(steps)
            and steps > 0
            and e["error"] is None,
            f"{path}: invalid native episode {key}",
        )
        rows[key] = (success, steps)
    require(
        sum(e["success"] for e in episodes) == group["successes"],
        f"{path}: incorrect native successes",
    )


def native_rows(path):
    data = json.loads(path.read_text())
    require(data["task_suite"] == "libero_10", f"{path}: expected LIBERO-10")
    rows = {}
    trials = data["num_trials_per_task"]
    require(integer(trials) and trials > 0, f"{path}: invalid trial count")
    for group in data["task_results"]:
        task = group["task_id"]
        require(integer(task) and task < 10, f"{path}: invalid task")
        episodes = group["episode_results"]
        require(
            len(episodes) == group["episodes"] == trials,
            f"{path}: incomplete task {task}",
        )
        add_native_episodes(path, task, group, rows, trials)
    require(
        rows
        and len(rows) == data["total_episodes"]
        and sum(r[0] for r in rows.values()) == data["total_successes"],
        f"{path}: native totals differ",
    )
    require(
        set(data["selected_task_ids"]) == {t for t, _ in rows},
        f"{path}: native task list differs",
    )
    return rows


def predicates(row, steps):
    result = {}
    for name, history in zip(row["goals"], row["transitions"], strict=True):
        require(
            isinstance(name, list) and name and all(isinstance(w, str) for w in name),
            "invalid replay goal",
        )
        goal = tuple(name)
        require(
            goal not in result and history and history[0][0] == 0,
            "duplicate goal or missing initial predicate",
        )
        previous = -1
        for step, value in history:
            require(
                integer(step) and previous < step <= steps and type(value) is bool,
                "invalid predicate transition",
            )
            previous = step
        result[goal] = (any(value for _, value in history), history[-1][1])
    require(result, "missing replay goals")
    return result


def verified_arm(native, replay):
    episodes = native_rows(native)
    rows = {}
    for line in replay.read_text().splitlines():
        r = json.loads(line)
        task, state = r["task"], r["state"]
        key = (task, state)
        require(
            integer(task) and integer(state) and key in episodes and key not in rows,
            f"{replay}: unexpected or duplicate {key}",
        )
        success, steps = episodes[key]
        for source in ("native", "replay"):
            count = r[source + "_steps"]
            require(
                r[source + "_success"] is success and integer(count) and count == steps,
                f"{replay}: {key} {source} outcome or steps differ",
            )
        require(
            r["matched"] is True and re.fullmatch("[0-9a-f]{64}", r["init_sha256"]),
            f"{replay}: {key} lacks a matched state",
        )
        goals = predicates(r, steps)
        require(
            all(at_end for _, at_end in goals.values()) == success,
            f"{replay}: {key} terminal goals contradict outcome",
        )
        if task == 3:
            require(set(goals) == {BOWL, DRAWER}, "unexpected drawer goal names")
        if task == 6:
            require(
                set(goals) == {MUG, PUDDING}
                and integer(r["pudding_pad_steps"])
                and r["pudding_pad_steps"] <= steps,
                "invalid pudding goals or contact count",
            )
        rows[key] = (success, r["init_sha256"], goals, r.get("pudding_pad_steps", 0))
    require(rows.keys() == episodes.keys(), f"{replay}: missing native episodes")
    return rows


def losing_stages(task, goals, contacts):
    if task == 3 and goals[BOWL][1] and not goals[DRAWER][1]:
        yield "bowl_inside_drawer_not_closed"
    if task == 6 and not goals[PUDDING][0]:
        if contacts > 0:
            yield "two_pad_contact_pudding_never_placed"
        if goals[MUG][1]:
            yield "mug_on_plate_pudding_never_placed"


def record_pair(tasks, total, key, control_row, sampler_row):
    task, state = key
    a, init_a, goals_a, pads_a = control_row
    b, init_b, goals_b, pads_b = sampler_row
    require(
        init_a == init_b and goals_a.keys() == goals_b.keys(),
        f"initial state or goal mismatch at {key}",
    )
    record = tasks.setdefault(
        task,
        {
            "counts": Counter(),
            "unmet_terminal_goals": {"gain": {}, "regression": {}},
            "losing_arm_states": {"gain": {}, "regression": {}},
        },
    )
    if a == b:
        outcome = "both_success" if a else "both_failure"
    else:
        outcome = "gain" if b else "regression"
    counts = {
        "pairs": 1,
        "control_success": int(a),
        "sampler_success": int(b),
        outcome: 1,
    }
    record["counts"].update(counts)
    total.update(counts)
    if a != b:
        for goal, (_, at_end) in (goals_a if b else goals_b).items():
            if not at_end:
                record["unmet_terminal_goals"][outcome].setdefault(
                    " ".join(goal), []
                ).append(state)
        failed = record["losing_arm_states"][outcome]
        for stage in losing_stages(
            task, goals_a if b else goals_b, pads_a if b else pads_b
        ):
            failed.setdefault(stage, []).append(state)


def compare(pairs):
    tasks, seen, total = {}, set(), Counter()
    for native_a, replay_a, native_b, replay_b in pairs:
        control = verified_arm(native_a, replay_a)
        sampler = verified_arm(native_b, replay_b)
        require(
            control.keys() == sampler.keys(), "native cohort keys differ between arms"
        )
        require(not seen.intersection(control), "cohort overlaps an earlier input pair")
        seen.update(control)
        for key in sorted(control):
            record_pair(tasks, total, key, control[key], sampler[key])
    return {"total": total, "tasks": tasks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        type=Path,
        action="append",
        nargs=4,
        required=True,
        metavar=(
            "CONTROL_NATIVE",
            "CONTROL_REPLAY",
            "SAMPLER_NATIVE",
            "SAMPLER_REPLAY",
        ),
    )
    args = parser.parse_args()
    try:
        report = compare(args.pair)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        parser.exit(1, f"invalid comparison: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
