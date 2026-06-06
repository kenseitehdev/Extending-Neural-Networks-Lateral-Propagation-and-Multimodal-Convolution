# file: sqlite_lateral_rnn.py

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TOK_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class Sample:
    label: int
    text: str
    token_ids: List[int]


@dataclass(frozen=True)
class BudgetResult:
    budget: int
    train_avg_loss: float
    train_avg_confidence: float
    train_acc: float
    test_loss: float
    test_confidence: float
    test_pred: int
    test_label: int
    test_correct: int


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return text.strip()


def tokenize(text: str) -> List[str]:
    return TOK_RE.findall(clean_text(text).lower())


def normalize_label(value: object) -> int | None:
    text = clean_text(value).lower()
    if text in {"positive", "pos", "1", "true", "yes"}:
        return 1
    if text in {"negative", "neg", "0", "false", "no"}:
        return 0
    try:
        return 1 if float(text) > 0 else 0
    except Exception:
        return None


def pick_column(fieldnames: Sequence[str], wanted: Sequence[str]) -> str | None:
    lower_map = {str(name).strip().lower(): name for name in fieldnames}
    for name in wanted:
        if name in lower_map:
            return lower_map[name]
    return None


def load_labeled_rows(csv_path: Path) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        text_col = pick_column(fieldnames, ["review", "text", "content", "comment", "comments"])
        label_col = pick_column(fieldnames, ["sentiment", "label", "target", "polarity", "class"])

        if text_col is None or label_col is None:
            if len(fieldnames) >= 2:
                text_col = fieldnames[0]
                label_col = fieldnames[1]
            else:
                raise ValueError("could not detect text and label columns")

        pos_rows: List[Tuple[int, str]] = []
        neg_rows: List[Tuple[int, str]] = []

        for row in reader:
            text = clean_text(row.get(text_col, ""))
            label = normalize_label(row.get(label_col, ""))
            if not text or label is None:
                continue
            if label == 1:
                pos_rows.append((1, text))
            else:
                neg_rows.append((0, text))

    return pos_rows, neg_rows


def split_rows(
    pos_rows: List[Tuple[int, str]],
    neg_rows: List[Tuple[int, str]],
    train_per_label: int,
    test_per_label: int,
    seed: int,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    rng = random.Random(seed)

    pos = list(pos_rows)
    neg = list(neg_rows)
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_train_n = min(train_per_label, len(pos))
    neg_train_n = min(train_per_label, len(neg))
    pos_test_n = min(test_per_label, max(0, len(pos) - pos_train_n))
    neg_test_n = min(test_per_label, max(0, len(neg) - neg_train_n))

    train_rows = pos[:pos_train_n] + neg[:neg_train_n]
    test_rows = pos[pos_train_n:pos_train_n + pos_test_n] + neg[neg_train_n:neg_train_n + neg_test_n]

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def build_vocab(train_rows: Sequence[Tuple[int, str]], max_vocab: int) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for _, text in train_rows:
        for tok in tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1

    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    stoi: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    limit = max(0, max_vocab - 2)

    for tok, _count in ordered[:limit]:
        if tok not in stoi:
            stoi[tok] = len(stoi)

    return stoi


def encode_text(stoi: Dict[str, int], text: str, max_len: int) -> List[int]:
    ids: List[int] = []
    for tok in tokenize(text):
        ids.append(stoi.get(tok, 1))
        if len(ids) >= max_len:
            break
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def build_samples(rows: Sequence[Tuple[int, str]], stoi: Dict[str, int], max_len: int) -> List[Sample]:
    return [Sample(label=label, text=text, token_ids=encode_text(stoi, text, max_len)) for label, text in rows]


def ensure_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS node_memory (
            ns TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            pos INTEGER NOT NULL,
            token_id INTEGER NOT NULL,
            weight REAL NOT NULL,
            cnt INTEGER NOT NULL,
            PRIMARY KEY (ns, class_id, pos, token_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS lateral_memory (
            ns TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            from_token INTEGER NOT NULL,
            to_token INTEGER NOT NULL,
            weight REAL NOT NULL,
            cnt INTEGER NOT NULL,
            PRIMARY KEY (ns, class_id, from_token, to_token)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS class_stats (
            ns TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            seen INTEGER NOT NULL,
            PRIMARY KEY (ns, class_id)
        )
        """
    )
    conn.commit()


def clear_namespace(conn: sqlite3.Connection, namespace: str) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM node_memory WHERE ns = ?", (namespace,))
    cur.execute("DELETE FROM lateral_memory WHERE ns = ?", (namespace,))
    cur.execute("DELETE FROM class_stats WHERE ns = ?", (namespace,))
    conn.commit()


def upsert_node(
    cur: sqlite3.Cursor,
    namespace: str,
    class_id: int,
    pos: int,
    token_id: int,
    delta: float,
) -> None:
    cur.execute(
        """
        INSERT INTO node_memory (ns, class_id, pos, token_id, weight, cnt)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(ns, class_id, pos, token_id)
        DO UPDATE SET
            weight = node_memory.weight + excluded.weight,
            cnt = node_memory.cnt + 1
        """,
        (namespace, class_id, pos, token_id, delta),
    )


def upsert_lateral(
    cur: sqlite3.Cursor,
    namespace: str,
    class_id: int,
    from_token: int,
    to_token: int,
    delta: float,
) -> None:
    cur.execute(
        """
        INSERT INTO lateral_memory (ns, class_id, from_token, to_token, weight, cnt)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(ns, class_id, from_token, to_token)
        DO UPDATE SET
            weight = lateral_memory.weight + excluded.weight,
            cnt = lateral_memory.cnt + 1
        """,
        (namespace, class_id, from_token, to_token, delta),
    )


def inc_class_seen(cur: sqlite3.Cursor, namespace: str, class_id: int) -> None:
    cur.execute(
        """
        INSERT INTO class_stats (ns, class_id, seen)
        VALUES (?, ?, 1)
        ON CONFLICT(ns, class_id)
        DO UPDATE SET seen = class_stats.seen + 1
        """,
        (namespace, class_id),
    )


def get_class_seen(cur: sqlite3.Cursor, namespace: str, classes: int) -> List[int]:
    counts = [0 for _ in range(classes)]
    cur.execute("SELECT class_id, seen FROM class_stats WHERE ns = ?", (namespace,))
    for class_id, seen in cur.fetchall():
        if 0 <= class_id < classes:
            counts[class_id] = int(seen)
    return counts


def train_sample(conn: sqlite3.Connection, namespace: str, sample: Sample) -> None:
    cur = conn.cursor()
    inc_class_seen(cur, namespace, sample.label)

    prev = 0
    for pos, tok in enumerate(sample.token_ids):
        if tok == 0:
            prev = tok
            continue

        node_delta = 1.0 / (1.0 + float(pos))
        upsert_node(cur, namespace, sample.label, pos, tok, node_delta)

        if prev != 0:
            upsert_lateral(cur, namespace, sample.label, prev, tok, 0.75)

        prev = tok

    conn.commit()


def recurrent_scores(conn: sqlite3.Connection, namespace: str, token_ids: Sequence[int], classes: int) -> List[float]:
    cur = conn.cursor()
    class_seen = get_class_seen(cur, namespace, classes)
    hidden = [0.0 for _ in range(classes)]

    prev = 0
    for pos, tok in enumerate(token_ids):
        local = [0.0 for _ in range(classes)]

        if tok != 0:
            cur.execute(
                "SELECT class_id, weight FROM node_memory WHERE ns = ? AND pos = ? AND token_id = ?",
                (namespace, pos, tok),
            )
            for class_id, weight in cur.fetchall():
                if 0 <= class_id < classes:
                    local[class_id] += float(weight)

            if prev != 0:
                cur.execute(
                    "SELECT class_id, weight FROM lateral_memory WHERE ns = ? AND from_token = ? AND to_token = ?",
                    (namespace, prev, tok),
                )
                for class_id, weight in cur.fetchall():
                    if 0 <= class_id < classes:
                        local[class_id] += 0.85 * float(weight)

        for class_id in range(classes):
            hidden[class_id] = 0.70 * hidden[class_id] + local[class_id]
            if class_seen[class_id] > 0:
                hidden[class_id] = hidden[class_id] / math.sqrt(float(class_seen[class_id]))

        prev = tok

    return hidden


def softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    denom = sum(exps)
    if denom <= 0.0:
        return [0.0 for _ in scores]
    return [x / denom for x in exps]


def predict(conn: sqlite3.Connection, namespace: str, sample: Sample, classes: int) -> Tuple[int, float, float, List[float]]:
    scores = recurrent_scores(conn, namespace, sample.token_ids, classes)
    probs = softmax(scores)
    pred = max(range(classes), key=lambda i: probs[i])
    confidence = probs[pred]
    loss = -math.log(probs[sample.label] + 1e-9)
    return pred, confidence, loss, probs


def run_budget(
    conn: sqlite3.Connection,
    namespace: str,
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    budget: int,
    classes: int,
    report_every: int,
    seed: int,
    test_per_run: int,
    reset_namespace: bool,
) -> BudgetResult:
    if reset_namespace:
        clear_namespace(conn, namespace)

    rng = random.Random(seed)
    total_loss = 0.0
    total_conf = 0.0
    hits = 0.0

    for step in range(1, budget + 1):
        sample = rng.choice(train_samples)
        train_sample(conn, namespace, sample)
        pred, conf, loss, _ = predict(conn, namespace, sample, classes)

        total_loss += loss
        total_conf += conf
        if pred == sample.label:
            hits += 1.0

        if step % report_every == 0 or step == budget:
            print(
                f"train_episodes {step} "
                f"avg_loss {total_loss / step:.6f} "
                f"avg_confidence {total_conf / step:.6f} "
                f"acc {hits / step:.6f}"
            )

    test_loss_sum = 0.0
    test_conf_sum = 0.0
    test_correct_sum = 0
    test_pred_last = 0
    test_label_last = 0

    for _ in range(test_per_run):
        sample = rng.choice(test_samples)
        pred, conf, loss, _ = predict(conn, namespace, sample, classes)
        test_loss_sum += loss
        test_conf_sum += conf
        test_correct_sum += int(pred == sample.label)
        test_pred_last = pred
        test_label_last = sample.label

    avg_test_loss = test_loss_sum / test_per_run
    avg_test_conf = test_conf_sum / test_per_run
    avg_test_correct = 1 if (test_correct_sum / test_per_run) >= 0.5 else 0

    print(
        f"test_loss {avg_test_loss:.6f} "
        f"test_confidence {avg_test_conf:.6f} "
        f"test_pred {test_pred_last} "
        f"test_label {test_label_last} "
        f"test_correct {avg_test_correct}"
    )

    return BudgetResult(
        budget=budget,
        train_avg_loss=total_loss / budget,
        train_avg_confidence=total_conf / budget,
        train_acc=hits / budget,
        test_loss=avg_test_loss,
        test_confidence=avg_test_conf,
        test_pred=test_pred_last,
        test_label=test_label_last,
        test_correct=avg_test_correct,
    )


def write_report(report_path: Path, results: Sequence[BudgetResult], vocab_size: int, max_len: int, seed: int, db_path: Path) -> None:
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "budget",
                "train_avg_loss",
                "train_avg_confidence",
                "train_acc",
                "test_loss",
                "test_confidence",
                "test_pred",
                "test_label",
                "test_correct",
                "vocab_size",
                "max_len",
                "seed",
                "db_path",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.budget,
                    row.train_avg_loss,
                    row.train_avg_confidence,
                    row.train_acc,
                    row.test_loss,
                    row.test_confidence,
                    row.test_pred,
                    row.test_label,
                    row.test_correct,
                    vocab_size,
                    max_len,
                    seed,
                    str(db_path),
                ]
            )


def parse_budgets(raw: str) -> List[int]:
    budgets: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            budgets.append(int(part))
    if not budgets:
        raise ValueError("budgets cannot be empty")
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite lateral-propagation text classifier")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--max-vocab", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--train-per-label", type=int, default=1000)
    parser.add_argument("--test-per-label", type=int, default=200)
    parser.add_argument("--budgets", type=str, default="500,1000,1500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--test-per-run", type=int, default=1)
    parser.add_argument("--db-name", type=str, default="sqlite_lateral_rnn.sqlite")
    parser.add_argument("--keep-memory-across-budgets", action="store_true")
    args = parser.parse_args()

    if not args.csv_path.is_file():
        raise SystemExit(f"csv not found: {args.csv_path}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_budgets(args.budgets)

    pos_rows, neg_rows = load_labeled_rows(args.csv_path)
    train_rows, test_rows = split_rows(
        pos_rows,
        neg_rows,
        train_per_label=args.train_per_label,
        test_per_label=args.test_per_label,
        seed=args.seed,
    )

    stoi = build_vocab(train_rows, args.max_vocab)
    train_samples = build_samples(train_rows, stoi, args.max_len)
    test_samples = build_samples(test_rows, stoi, args.max_len)

    print(f"train_rows {len(train_samples)} test_rows {len(test_samples)}")
    print(
        f"vocab_size {len(stoi)} max_vocab {args.max_vocab} "
        f"max_len {args.max_len} train_per_label {args.train_per_label} "
        f"test_per_label {args.test_per_label} seed {args.seed}"
    )

    db_path = args.work_dir / args.db_name
    conn = sqlite3.connect(db_path)
    ensure_db(conn)

    results: List[BudgetResult] = []
    for budget in budgets:
        print("==============================")
        print(f"run_budget {budget}")
        namespace = f"run_{budget}_{args.seed}"
        result = run_budget(
            conn=conn,
            namespace=namespace,
            train_samples=train_samples,
            test_samples=test_samples,
            budget=budget,
            classes=args.classes,
            report_every=args.report_every,
            seed=args.seed + budget,
            test_per_run=args.test_per_run,
            reset_namespace=not args.keep_memory_across_budgets,
        )
        results.append(result)

    conn.close()

    report_path = args.work_dir / "sqlite_lateral_rnn_runs.csv"
    write_report(report_path, results, vocab_size=len(stoi), max_len=args.max_len, seed=args.seed, db_path=db_path)
    print(f"report_csv {report_path}")
    print(f"sqlite_db {db_path}")


if __name__ == "__main__":
    main()