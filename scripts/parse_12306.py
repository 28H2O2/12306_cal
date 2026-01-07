#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12306 txt 邮件导出解析器（可复现）。

输入：一个目录下的若干 *.txt（或显式指定多个 txt）
输出：
  - events.jsonl   每封票务邮件 1 条事件记录（含 tickets[]）
  - tickets.json   展平后的票面级记录（便于统计/筛选）
  - metadata.json  运行元信息（版本、输入摘要等）
  - report.html    单文件离线报表（可选开关，默认生成）

设计目标：
  - 仅使用 Python 标准库（离线可跑）
  - 面向 12306 通知邮件的多年代模板（购票/退票/改签/候补兑现）
  - 自动过滤非 12306/非票务邮件（例如 Steam、账号激活等）
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PARSER_VERSION = "0.1.0"


EVENT_PURCHASE = "purchase"
EVENT_REFUND = "refund"
EVENT_RESCHEDULE = "reschedule"
EVENT_STANDBY_FULFILLED = "standby_fulfilled"
EVENT_STATION_CHANGE = "station_change"  # 预留


TRAIN_PREFIXES = {"K", "T", "Z", "D", "C", "G", "Y", "S", "L"}


@dataclasses.dataclass(frozen=True)
class SourceRef:
    file: str
    start_line: int
    end_line: int


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    # 12306 导出的 txt 通常是 utf-8；为了兼容异常字符，使用 errors=ignore。
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_mail_blocks(text: str, src_file: str) -> Iterable[Tuple[str, SourceRef]]:
    """
    基于“正文:”与“###”分隔符做流式切块。
    """
    lines = text.splitlines()
    in_body = False
    buf: List[str] = []
    start_line = 0
    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if not in_body:
            if s == "正文:":
                in_body = True
                buf = []
                start_line = idx
            continue

        # in_body == True
        if s == "###":
            body = "\n".join(buf).strip()
            if body:
                yield body, SourceRef(file=src_file, start_line=start_line, end_line=idx)
            in_body = False
            buf = []
            start_line = 0
            continue

        buf.append(line)

    # EOF
    if in_body:
        body = "\n".join(buf).strip()
        if body:
            yield body, SourceRef(file=src_file, start_line=start_line, end_line=len(lines))


def _normalize_spaces(s: str) -> str:
    # 合并多空格，便于 regex
    return re.sub(r"\s+", " ", s).strip()


def _detect_event_type(body: str) -> Optional[str]:
    b = body
    if "12306通知邮件" not in b:
        return None

    # station_change（变更到站/变更车票）
    # 注意：温馨提示里会出现“改签、变更到站、退票…”，不能据此误判。
    if ("成功变更车票" in b or "成功办理了变更到站" in b or "成功变更到站" in b) and (
        "变更后的车票信息如下" in b or "变更到站后的车票信息如下" in b
    ):
        return EVENT_STATION_CHANGE

    # 注意：有些邮件没有“您于YYYY年…”，例如候补兑现。
    # purchase
    if ("成功购买" in b) and ("所购车票信息" in b or "车票信息如下" in b):
        if "所购车票信息" in b:
            return EVENT_PURCHASE
        # 兜底：某些模板可能写“车票信息如下”但仍是购买
        if "票款共计" in b and "订单号码" in b:
            return EVENT_PURCHASE

    # refund
    if ("成功办理了退票业务" in b) and ("所退车票信息" in b):
        return EVENT_REFUND

    # reschedule
    if ("成功改签车票" in b) and ("改签后的车票信息" in b):
        return EVENT_RESCHEDULE

    # standby_fulfilled
    if ("候补购票" in b) and ("成功兑现" in b) and ("车票信息如下" in b):
        return EVENT_STANDBY_FULFILLED

    return None


def _extract_order_id(body: str) -> Optional[str]:
    m = re.search(r"订单号码\s*([A-Za-z0-9]+)", body)
    return m.group(1) if m else None


def _extract_first_date(body: str) -> Optional[str]:
    """
    解析“您于YYYY年MM月DD日”。
    返回 YYYY-MM-DD。
    """
    m = re.search(r"您于(\d{4})年(\d{2})月(\d{2})日", body)
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2), m.group(3)
    return f"{y}-{mo}-{d}"


def _extract_last_date(body: str) -> Optional[str]:
    """
    解析邮件落款日期：取全文最后一个“YYYY年MM月DD日”。
    返回 YYYY-MM-DD。
    """
    matches = list(re.finditer(r"(\d{4})年(\d{2})月(\d{2})日", body))
    if not matches:
        return None
    m = matches[-1]
    y, mo, d = m.group(1), m.group(2), m.group(3)
    return f"{y}-{mo}-{d}"


def _extract_amount(body: str, event_type: str) -> Optional[float]:
    patterns: List[str] = []
    if event_type in (EVENT_PURCHASE, EVENT_STANDBY_FULFILLED):
        patterns = [
            r"票款共计\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        ]
    elif event_type == EVENT_REFUND:
        patterns = [
            r"应退票款\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        ]
    elif event_type == EVENT_RESCHEDULE:
        patterns = [
            r"新车票票款共计\s*([0-9]+(?:\.[0-9]+)?)\s*元",
            r"新车票票款共计\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        ]
    elif event_type == EVENT_STATION_CHANGE:
        patterns = [
            r"新车票票款共计\s*([0-9]+(?:\.[0-9]+)?)\s*元",
            r"票款共计\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        ]
    else:
        patterns = [r"票款共计\s*([0-9]+(?:\.[0-9]+)?)\s*元"]

    for p in patterns:
        m = re.search(p, body)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _cut_ticket_chunk(body: str, event_type: str) -> Optional[str]:
    b = _normalize_spaces(body)

    intros: List[str] = []
    if event_type == EVENT_PURCHASE:
        intros = ["所购车票信息如下", "车票信息如下"]
    elif event_type == EVENT_REFUND:
        intros = ["所退车票信息如下"]
    elif event_type == EVENT_RESCHEDULE:
        intros = ["改签后的车票信息如下"]
    elif event_type == EVENT_STANDBY_FULFILLED:
        intros = ["车票信息如下"]
    elif event_type == EVENT_STATION_CHANGE:
        intros = ["变更后的车票信息如下", "变更到站后的车票信息如下", "车票信息如下"]
    else:
        intros = ["车票信息如下"]

    start_idx = -1
    intro_used = None
    for intro in intros:
        idx = b.find(intro)
        if idx != -1:
            start_idx = idx + len(intro)
            intro_used = intro
            break

    if start_idx == -1:
        return None

    chunk = b[start_idx:].lstrip("：: ").strip()
    if not chunk:
        return None

    end_markers = [
        "温馨提示",
        "为了确保旅客人身安全",
        "按购票时所使用在线支付工具",
        "按购票时所使用在线支付账工具",
        "感谢您使用中国铁路客户服务中心网站",
        "本邮件由系统自动发出",
    ]
    end_pos = None
    for m in end_markers:
        pos = chunk.find(m)
        if pos != -1:
            end_pos = pos if end_pos is None else min(end_pos, pos)
    if end_pos is not None:
        chunk = chunk[:end_pos].strip()

    # 有些 intro 之后直接跟着其他说明，导致 chunk 过短；仍返回由上层判断。
    if not chunk:
        return None
    return chunk


def _split_ticket_entries(ticket_chunk: str) -> List[str]:
    """
    优先按 1./2./3. 分割；没有编号则尽量取第一句作为一条票信息。
    """
    c = ticket_chunk.strip()
    if not c:
        return []

    # 统一中文句号
    c = c.replace("\u3002", "。")

    starts = [m.start() for m in re.finditer(r"\b\d+\.\s*", c)]
    if starts:
        starts.append(len(c))
        entries: List[str] = []
        for i in range(len(starts) - 1):
            seg = c[starts[i] : starts[i + 1]].strip()
            if seg:
                entries.append(seg)
        return entries

    # 无编号：多数情况下只有一张票，且在第一个句号结束
    first = c
    dot = first.find("。")
    if dot != -1:
        first = first[: dot + 1].strip()
    return [first] if first else []


def _parse_depart_datetime(token: str) -> Optional[str]:
    m = re.search(r"(\d{4})年(\d{2})月(\d{2})日(\d{2}):(\d{2})开", token)
    if not m:
        return None
    y, mo, d, hh, mm = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    return f"{y}-{mo}-{d}T{hh}:{mm}"


def _parse_route(token: str) -> Tuple[Optional[str], Optional[str]]:
    # 可能用“-”或“－”
    if "-" not in token and "－" not in token:
        return None, None
    t = token.replace("－", "-")
    parts = [p.strip() for p in t.split("-", 1)]
    if len(parts) != 2:
        return None, None
    return parts[0] or None, parts[1] or None


def _parse_train_no(token: str) -> Optional[str]:
    m = re.search(r"([A-Za-z]+\d+|\d+)\s*次列车", token)
    if not m:
        return None
    return m.group(1).upper()


def _train_prefix(train_no: Optional[str]) -> str:
    if not train_no:
        return "OTHER"
    if train_no[0].isalpha():
        p = train_no[0].upper()
        return p if p in TRAIN_PREFIXES else "OTHER"
    if train_no[0].isdigit():
        return "NUM"
    return "OTHER"


def _parse_price(token: str) -> Optional[float]:
    m = re.search(r"票价\s*([0-9]+(?:\.[0-9]+)?)\s*元", token)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_refund_fee(token: str) -> Optional[float]:
    m = re.search(r"退票费\s*([0-9]+(?:\.[0-9]+)?)\s*元", token)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_refund_amount(token: str) -> Optional[float]:
    m = re.search(r"应退票款\s*([0-9]+(?:\.[0-9]+)?)\s*元", token)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


KNOWN_SEAT_HINTS = [
    "商务座",
    "特等座",
    "一等座",
    "二等座",
    "优选一等",
    "优选二等",
    "一等软座",
    "二等软座",
    "硬座",
    "无座",
    "硬卧",
    "软卧",
    "包厢硬卧",
    "二等卧",
    "动卧",
    "软卧代二等座",
    "硬卧代硬座",
]


def _is_ticket_category(token: str) -> bool:
    # 票种：成人票/学生票/儿童票等
    if token in ("电子客票", "电子车票"):
        return False
    return token.endswith("票") and ("检票口" not in token) and ("票价" not in token) and ("退票费" not in token)


def _is_seat_type(token: str) -> bool:
    if token in ("电子客票", "电子车票"):
        return False
    # “N车无座”本质是无座席别的写法，视为席别
    if re.search(r"\b\d+\s*车\s*无座\b", token):
        return True
    if any(h in token for h in KNOWN_SEAT_HINTS):
        return True
    # 泛化：含“座/卧”但不是“票价/检票口/成人票…”
    if ("座" in token or "卧" in token) and (not _is_ticket_category(token)) and ("票价" not in token) and ("检票口" not in token):
        return True
    return False


def _parse_seat_raw(token: str) -> Optional[str]:
    # 通用座位/铺位 token：包含“车”且包含“号/铺/座/上铺/中铺/下铺”中的某个
    # 特判：无座经常写成“X车无座”，这不是一个有意义的座位号，统一归类为 seatType=无座，seatRaw 留空
    if re.search(r"\b\d+\s*车\s*无座\b", token):
        return None
    if "车" not in token:
        return None
    if any(x in token for x in ["号", "铺"]):
        return token
    # 有些是 “2F号”
    if re.search(r"\b\d+[A-Za-z]\s*号\b", token):
        return token
    return None


def _split_seat_fields(seat_raw: Optional[str]) -> Dict[str, Optional[str]]:
    """
    尝试从 seatRaw 中拆出 coachNo/seatNo/berthType（失败则返回 None）。
    """
    if not seat_raw:
        return {"coachNo": None, "seatNo": None, "berthType": None, "isExtraCoach": None}

    raw = seat_raw.strip()
    raw_norm = raw.replace("號", "号")

    is_extra = raw_norm.startswith("加")
    coach_no = None
    seat_no = None
    berth_type = None

    m = re.search(r"(加)?\s*(\d+)\s*车", raw_norm)
    if m:
        coach_no = (("加" + m.group(2)) if m.group(1) else m.group(2))

    # 上/中/下铺
    if "上铺" in raw_norm:
        berth_type = "上铺"
    elif "中铺" in raw_norm:
        berth_type = "中铺"
    elif "下铺" in raw_norm:
        berth_type = "下铺"

    # seatNo：优先抓“车”之后的内容
    m2 = re.search(r"车\s*(.+)$", raw_norm)
    if m2:
        seat_no = m2.group(1).strip()
    else:
        seat_no = raw_norm

    return {
        "coachNo": coach_no,
        "seatNo": seat_no,
        "berthType": berth_type,
        "isExtraCoach": bool(is_extra),
    }


def parse_ticket_entry(entry: str) -> Dict[str, Any]:
    """
    将一条票面描述解析为结构化字段。
    """
    e = entry.strip()
    e = e.replace("\u3002", "。")
    e = e.strip(" 。")

    # 去掉开头编号
    e = re.sub(r"^\s*\d+\.\s*", "", e)

    # 分词：兼容中文/英文逗号
    tokens = [t.strip() for t in re.split(r"[，,]", e) if t.strip()]

    passenger = tokens[0] if tokens else None

    depart_dt = None
    from_station = None
    to_station = None
    train_no = None
    seat_raw = None
    seat_type = None
    ticket_category = None
    price = None
    check_in_gate = None
    refund_fee = None
    refund_amount = None

    # 一次扫描：先抓强特征字段
    for t in tokens[1:]:
        if depart_dt is None:
            depart_dt = _parse_depart_datetime(t) or depart_dt
        if from_station is None and to_station is None:
            fs, ts = _parse_route(t)
            if fs and ts:
                from_station, to_station = fs, ts
        if train_no is None:
            train_no = _parse_train_no(t) or train_no

    # 二次扫描：剩余字段
    for t in tokens[1:]:
        if t.startswith("检票口"):
            gate = t[len("检票口") :].strip()
            check_in_gate = gate or check_in_gate
            continue
        if "票价" in t and price is None:
            price = _parse_price(t) or price
            continue
        if "退票费" in t and refund_fee is None:
            refund_fee = _parse_refund_fee(t) or refund_fee
            continue
        if "应退票款" in t and refund_amount is None:
            refund_amount = _parse_refund_amount(t) or refund_amount
            continue
        if seat_raw is None:
            sr = _parse_seat_raw(t)
            if sr:
                seat_raw = sr
                continue

    for t in tokens[1:]:
        if ticket_category is None and _is_ticket_category(t):
            ticket_category = t
            continue
        if seat_type is None and _is_seat_type(t):
            # 归一化：把 “N车无座” 统一为 “无座”
            if re.search(r"\b\d+\s*车\s*无座\b", t):
                seat_type = "无座"
                # 避免把 “N车无座” 当成 seatRaw
                if seat_raw and ("无座" in seat_raw):
                    seat_raw = None
            else:
                seat_type = t
            continue

    # 最终兜底：若 seatType 仍是 “N车无座” 或 seatRaw 含“无座”，统一为无座并清空 seatRaw
    if seat_type and re.search(r"\b\d+\s*车\s*无座\b", seat_type):
        seat_type = "无座"
    if seat_type == "无座" and seat_raw and ("无座" in seat_raw):
        seat_raw = None

    seat_fields = _split_seat_fields(seat_raw)
    prefix = _train_prefix(train_no)

    price_fen = None
    if price is not None:
        price_fen = int(round(price * 100))

    return {
        "passengerName": passenger,
        "departDateTime": depart_dt,
        "fromStation": from_station,
        "toStation": to_station,
        "trainNo": train_no,
        "trainTypePrefix": prefix,
        "seatType": seat_type,
        "ticketCategory": ticket_category,
        "seatRaw": seat_raw,
        **seat_fields,
        "priceYuan": price,
        "priceFen": price_fen,
        "checkInGate": check_in_gate,
        "refundFeeYuan": refund_fee,
        "refundAmountYuan": refund_amount,
    }


def _event_time_key(event: Dict[str, Any]) -> str:
    # 仅用于排序：优先 transactionDate，否则 mailDate，否则空
    return (event.get("transactionDate") or event.get("mailDate") or "") + "T00:00"


def _make_ticket_id(order_id: Optional[str], event_type: str, idx: int, src: SourceRef) -> str:
    base = f"{order_id or 'NOORDER'}|{event_type}|{idx}|{src.file}|{src.start_line}|{src.end_line}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def parse_events_from_files(input_files: List[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    tickets: List[Dict[str, Any]] = []

    for p in input_files:
        text = _read_text(p)
        for body, src in _iter_mail_blocks(text, src_file=str(p)):
            event_type = _detect_event_type(body)
            if not event_type:
                continue

            order_id = _extract_order_id(body)
            transaction_date = _extract_first_date(body)
            mail_date = _extract_last_date(body)
            total_amount = _extract_amount(body, event_type)

            ticket_chunk = _cut_ticket_chunk(body, event_type)
            entry_texts = _split_ticket_entries(ticket_chunk) if ticket_chunk else []

            parsed_tickets: List[Dict[str, Any]] = []
            for i, entry in enumerate(entry_texts):
                t = parse_ticket_entry(entry)
                t_id = _make_ticket_id(order_id, event_type, i, src)
                t.update(
                    {
                        "id": t_id,
                        "eventType": event_type,
                        "orderId": order_id,
                        "transactionDate": transaction_date,
                        "mailDate": mail_date,
                        "sourceFile": src.file,
                        "sourceStartLine": src.start_line,
                        "sourceEndLine": src.end_line,
                    }
                )
                parsed_tickets.append(t)
                tickets.append(t)

            event_id = hashlib.sha1(
                f"{event_type}|{order_id}|{src.file}|{src.start_line}|{src.end_line}".encode("utf-8")
            ).hexdigest()
            event = {
                "id": event_id,
                "eventType": event_type,
                "orderId": order_id,
                "transactionDate": transaction_date,
                "mailDate": mail_date,
                "totalAmountYuan": total_amount,
                "sourceFile": src.file,
                "sourceStartLine": src.start_line,
                "sourceEndLine": src.end_line,
                "tickets": parsed_tickets,
            }
            events.append(event)

    # 排序保证复现一致
    events.sort(key=_event_time_key)
    tickets.sort(key=lambda t: (t.get("departDateTime") or "", t.get("orderId") or "", t["id"]))

    return events, tickets


def _ticket_match_key(t: Dict[str, Any]) -> Tuple:
    return (
        t.get("orderId"),
        t.get("passengerName"),
        t.get("departDateTime"),
        t.get("fromStation"),
        t.get("toStation"),
        t.get("trainNo"),
        t.get("seatType"),
        t.get("priceYuan"),
    )


def label_ticket_status(events: List[Dict[str, Any]], tickets: List[Dict[str, Any]]) -> None:
    """
    在 tickets 上就地写入 status：active/refunded/rescheduled/unknown。
    规则（面向 all_events 口径）：
      - refund 事件的 tickets：refunded
      - purchase/standby_fulfilled/reschedule 默认 active
      - 若存在对应 refund 事件，匹配到的非 refund 票也标记为 refunded
      - 若同一 orderId+passenger 出现 reschedule，则早于改签的新旧购票票标记为 rescheduled（若未被 refunded 覆盖）
    """
    # 先默认
    active_event_types = {EVENT_PURCHASE, EVENT_STANDBY_FULFILLED, EVENT_RESCHEDULE, EVENT_STATION_CHANGE}
    change_event_types = {EVENT_RESCHEDULE, EVENT_STATION_CHANGE}

    for t in tickets:
        if t["eventType"] == EVENT_REFUND:
            t["status"] = "refunded"
        elif t["eventType"] in active_event_types:
            t["status"] = "active"
        else:
            t["status"] = "unknown"

    # refund 对应 purchase/reschedule 的联动标记
    refund_keys = set()
    for t in tickets:
        if t["eventType"] == EVENT_REFUND:
            refund_keys.add(_ticket_match_key(t))
    for t in tickets:
        if t["eventType"] != EVENT_REFUND and _ticket_match_key(t) in refund_keys:
            t["status"] = "refunded"

    # 改签/变更：标记同订单同乘车人中“被替换”的购票为 rescheduled
    # 1) 先收集每个 orderId+passenger 的“新票 key 集合”（来自 reschedule/station_change）
    changed_new_keys: Dict[Tuple[Optional[str], Optional[str]], set] = {}
    # 2) 同时记录最早的变更日期（若可用）
    change_min_date: Dict[Tuple[Optional[str], Optional[str]], str] = {}
    for ev in events:
        if ev["eventType"] not in change_event_types:
            continue
        ev_date = ev.get("transactionDate") or ev.get("mailDate") or ""
        for tk in ev.get("tickets", []):
            key = (ev.get("orderId"), tk.get("passengerName"))
            changed_new_keys.setdefault(key, set()).add(_ticket_match_key(tk))
            if ev_date:
                cur = change_min_date.get(key)
                if cur is None or ev_date < cur:
                    change_min_date[key] = ev_date

    for t in tickets:
        if t["eventType"] != EVENT_PURCHASE:
            continue
        key = (t.get("orderId"), t.get("passengerName"))
        new_keys = changed_new_keys.get(key)
        if not new_keys:
            continue

        # 若 purchase 票与任何“新票”不相同，则视为被替换（改签/变更），标记 rescheduled。
        # 若能拿到日期，则仅在 purchase 日期 <= 变更日期时才标记；否则直接标记（更鲁棒，处理同日改签）。
        if _ticket_match_key(t) in new_keys:
            continue
        if t.get("status") == "refunded":
            continue

        ev_date = t.get("transactionDate") or t.get("mailDate") or ""
        cutoff = change_min_date.get(key)
        if cutoff and ev_date and ev_date > cutoff:
            continue
        t["status"] = "rescheduled"


def _format_yuan(amount: Optional[float]) -> str:
    if amount is None:
        return "-"
    # 保留 2 位，去掉多余 0
    s = f"{amount:.2f}"
    if s.endswith("00"):
        return s[:-3]
    if s.endswith("0"):
        return s[:-1]
    return s


def build_report_html(tickets: List[Dict[str, Any]], events: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    # 为减小体积：report 只需要 tickets；events 用于统计“事件数量/金额”会更准确，但也可以只传汇总。
    event_counts: Dict[str, int] = {}
    amount_purchase = 0.0
    amount_refund = 0.0
    for ev in events:
        et = ev.get("eventType") or "unknown"
        event_counts[et] = event_counts.get(et, 0) + 1
        amt = ev.get("totalAmountYuan")
        if isinstance(amt, (int, float)):
            if et in (EVENT_PURCHASE, EVENT_STANDBY_FULFILLED, EVENT_RESCHEDULE, EVENT_STATION_CHANGE):
                amount_purchase += float(amt)
            elif et == EVENT_REFUND:
                amount_refund += float(amt)

    summary = {
        "eventCounts": event_counts,
        "amountPurchaseYuan": round(amount_purchase, 2),
        "amountRefundYuan": round(amount_refund, 2),
        "netSpendYuan": round(amount_purchase - amount_refund, 2),
    }

    data_obj = {
        "metadata": metadata,
        "summary": summary,
        "tickets": tickets,
    }

    data_json = json.dumps(data_obj, ensure_ascii=False, separators=(",", ":"))

    # 单文件 HTML（原生 JS + 简单 CSS）
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>12306购票记录统计</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #111a2e;
      --muted: #8aa0c7;
      --text: #e7eeff;
      --line: rgba(255,255,255,.10);
      --accent: #4ea1ff;
      --good: #3ddc97;
      --bad: #ff6b6b;
      --warn: #ffd166;
      --chip: rgba(78,161,255,.15);
      --shadow: rgba(0,0,0,.35);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif;
    }}
    body {{
      margin: 0;
      background: radial-gradient(1200px 600px at 20% 10%, rgba(78,161,255,.25), transparent 60%),
                  radial-gradient(900px 500px at 80% 20%, rgba(61,220,151,.18), transparent 55%),
                  var(--bg);
      color: var(--text);
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 20px; margin: 0 0 6px; }}
    .sub {{ color: var(--muted); font-size: 12px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 12px 28px var(--shadow);
      padding: 14px;
    }}
    .kpi {{ display:flex; align-items:baseline; gap: 10px; }}
    .kpi .v {{ font-size: 22px; font-weight: 700; letter-spacing: .2px; }}
    .kpi .l {{ color: var(--muted); font-size: 12px; }}
    .row {{ display:flex; gap: 10px; flex-wrap: wrap; }}
    .chip {{
      background: var(--chip);
      border: 1px solid rgba(78,161,255,.25);
      color: #cfe4ff;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 12px;
      line-height: 20px;
    }}
    .filters .field {{
      display:flex; flex-direction: column; gap: 6px;
    }}
    label {{ font-size: 12px; color: var(--muted); }}
    input, select {{
      background: rgba(0,0,0,.25);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 10px;
      padding: 8px 10px;
      outline: none;
    }}
    input::placeholder {{ color: rgba(231,238,255,.45); }}
    .btn {{
      background: rgba(78,161,255,.15);
      border: 1px solid rgba(78,161,255,.35);
      color: #d7eaff;
      border-radius: 10px;
      padding: 9px 12px;
      cursor: pointer;
    }}
    .btn:hover {{ background: rgba(78,161,255,.22); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
      border: 1px solid var(--line);
    }}
    th, td {{
      text-align: left;
      padding: 10px 10px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      vertical-align: top;
    }}
    th {{ color: #cfe0ff; font-weight: 650; background: rgba(0,0,0,.18); position: sticky; top: 0; }}
    tr:hover td {{ background: rgba(255,255,255,.03); }}
    .status {{
      display:inline-flex; align-items:center; gap: 6px;
      font-size: 12px; padding: 2px 10px; border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(0,0,0,.2);
    }}
    .status.active {{ border-color: rgba(61,220,151,.35); color: #b7ffe3; background: rgba(61,220,151,.10); }}
    .status.refunded {{ border-color: rgba(255,107,107,.35); color: #ffd0d0; background: rgba(255,107,107,.10); }}
    .status.rescheduled {{ border-color: rgba(255,209,102,.35); color: #ffe7b3; background: rgba(255,209,102,.12); }}
    .muted {{ color: var(--muted); }}
    .bars {{ display:flex; flex-direction:column; gap: 8px; }}
    .barrow {{ display:grid; grid-template-columns: 120px 1fr 70px; gap: 10px; align-items:center; }}
    .bar {{
      height: 10px; border-radius: 999px; background: rgba(255,255,255,.08);
      border: 1px solid var(--line);
      overflow:hidden;
    }}
    .bar > i {{ display:block; height: 100%; width: 0%; background: linear-gradient(90deg, rgba(78,161,255,.95), rgba(61,220,151,.85)); }}
    .split {{ display:flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    details summary {{ cursor: pointer; color: #cfe4ff; }}
    .small {{ font-size: 12px; color: var(--muted); }}
    .route {{ white-space: nowrap; word-break: keep-all; overflow-wrap: normal; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>12306购票记录统计（离线报表）</h1>
    <div class="sub" id="metaLine"></div>

    <div class="grid">
      <div class="card" style="grid-column: span 12;">
        <div class="row" id="eventChips"></div>
      </div>

      <div class="card" style="grid-column: span 4;">
        <div class="kpi"><div class="v" id="kpiTickets">-</div><div class="l">票记录（当前筛选后）</div></div>
        <div class="kpi" style="margin-top:10px"><div class="v" id="kpiTrips">-</div><div class="l">趟次（按 车次+开车时间+区间 去重）</div></div>
      </div>
      <div class="card" style="grid-column: span 4;">
        <div class="kpi"><div class="v" id="kpiSpend">-</div><div class="l">票价合计（当前筛选后）</div></div>
        <div class="kpi" style="margin-top:10px"><div class="v" id="kpiNet">-</div><div class="l">净支出（全量 events 口径）</div></div>
      </div>
      <div class="card" style="grid-column: span 4;">
        <div class="kpi"><div class="v" id="kpiStations">-</div><div class="l">涉及车站数（当前筛选后）</div></div>
        <div class="kpi" style="margin-top:10px"><div class="v" id="kpiTrains">-</div><div class="l">涉及车次（当前筛选后）</div></div>
      </div>

      <div class="card filters" style="grid-column: span 12;">
        <div class="split">
          <div>
            <div style="font-weight:650">筛选（默认按开车时间）</div>
            <div class="small">支持年/月 + 自定义日期范围 + 车站/车次/价格/席别/状态</div>
          </div>
          <button class="btn" id="btnReset">重置筛选</button>
        </div>
        <div class="grid" style="margin-top:12px">
          <div class="field" style="grid-column: span 2;">
            <label>乘车人</label>
            <select id="fPassenger"></select>
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>年份</label>
            <select id="fYear"></select>
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>月份</label>
            <select id="fMonth"></select>
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>开始日期</label>
            <input id="fStart" type="date" />
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>结束日期</label>
            <input id="fEnd" type="date" />
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>车次（包含）</label>
            <input id="fTrain" placeholder="如 G1451 / D9" />
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>车站（发/到包含）</label>
            <input id="fStation" placeholder="如 杭州东 / 北京南" />
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>价格 ≥（元）</label>
            <input id="fMinPrice" type="number" step="0.5" placeholder="0" />
          </div>
          <div class="field" style="grid-column: span 2;">
            <label>价格 ≤（元）</label>
            <input id="fMaxPrice" type="number" step="0.5" placeholder="9999" />
          </div>
          <div class="field" style="grid-column: span 3;">
            <label>席别</label>
            <select id="fSeat"></select>
          </div>
          <div class="field" style="grid-column: span 3;">
            <label>状态</label>
            <select id="fStatus">
              <option value="all">全部</option>
              <option value="active">未退改（active）</option>
              <option value="refunded">已退票（refunded）</option>
              <option value="rescheduled">已改签旧票（rescheduled）</option>
              <option value="unknown">未知</option>
            </select>
          </div>
        </div>
      </div>

      <div class="card" style="grid-column: span 6;">
        <div style="font-weight:650; margin-bottom:10px">车次前缀比例（KTZDCGYSL/NUM/OTHER）</div>
        <div class="bars" id="chartPrefix"></div>
      </div>
      <div class="card" style="grid-column: span 6;">
        <div style="font-weight:650; margin-bottom:10px">席别分布（TOP10）</div>
        <div class="bars" id="chartSeat"></div>
      </div>

      <div class="card" style="grid-column: span 4;">
        <div style="font-weight:650; margin-bottom:10px">车站 TOP10（发+到合计）</div>
        <div class="bars" id="topStations"></div>
      </div>
      <div class="card" style="grid-column: span 4;">
        <div style="font-weight:650; margin-bottom:10px">车次 TOP10</div>
        <div class="bars" id="topTrains"></div>
      </div>
      <div class="card" style="grid-column: span 4;">
        <div style="font-weight:650; margin-bottom:10px">区间 TOP10</div>
        <div class="bars" id="topSegments"></div>
      </div>

      <div class="card" style="grid-column: span 12;">
        <div class="split">
          <div style="font-weight:650">明细（可筛选/可展开查看订单号等）</div>
          <div class="small" id="tableHint"></div>
        </div>
        <div style="max-height: 520px; overflow:auto; margin-top:10px;">
          <table>
            <thead>
              <tr>
                <th style="width:140px">开车时间</th>
                <th style="width:140px">车次</th>
                <th style="min-width:220px">区间</th>
                <th style="width:110px">乘车人</th>
                <th style="width:140px">席别/座位</th>
                <th style="width:90px">票价</th>
                <th style="width:120px">状态</th>
                <th style="width:220px">更多</th>
              </tr>
            </thead>
            <tbody id="tbody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script id="data" type="application/json">{data_json}</script>
  <script>
  const DATA = JSON.parse(document.getElementById('data').textContent);
  const ticketsAll = DATA.tickets || [];

  const fmtYuan = (n) => {{
    if (n === null || n === undefined || Number.isNaN(n)) return '-';
    const s = (Math.round(n * 100) / 100).toFixed(2);
    if (s.endsWith('00')) return s.slice(0, -3);
    if (s.endsWith('0')) return s.slice(0, -1);
    return s;
  }};

  const safe = (s) => (s === null || s === undefined) ? '' : String(s);

  const eventLabel = (k) => {{
    const map = {{
      purchase: '购票',
      refund: '退票',
      reschedule: '改签',
      standby_fulfilled: '候补兑现',
      station_change: '变更车票',
      unknown: '未知',
    }};
    return map[k] || k;
  }};

  const statusLabel = (k) => {{
    const map = {{
      active: '未退改',
      refunded: '已退票',
      rescheduled: '已改签旧票',
      unknown: '未知',
    }};
    return map[k] || k;
  }};

  const getFileName = (path) => {{
    if (!path) return '-';
    // 处理 Unix 和 Windows 路径分隔符
    const parts = path.replace(/\\\\/g, '/').split('/');
    return parts[parts.length - 1];
  }};

  function parseDepartDate(t) {{
    // departDateTime: YYYY-MM-DDTHH:mm
    const dt = t.departDateTime;
    if (!dt || dt.length < 10) return null;
    const d = dt.slice(0, 10);
    return d; // YYYY-MM-DD
  }}

  function yearOf(t) {{
    const d = parseDepartDate(t);
    return d ? d.slice(0, 4) : null;
  }}
  function monthOf(t) {{
    const d = parseDepartDate(t);
    return d ? d.slice(5, 7) : null;
  }}

  function uniq(arr) {{
    return [...new Set(arr)];
  }}

  function computeSeatOptions(ts) {{
    const seats = uniq(ts.map(t => t.seatType).filter(Boolean)).sort((a,b) => a.localeCompare(b,'zh-CN'));
    return ['全部', ...seats];
  }}

  function computePassengerOptions(ts) {{
    const names = uniq(ts.map(t => t.passengerName).filter(Boolean)).sort((a,b) => a.localeCompare(b,'zh-CN'));
    return ['全部', ...names];
  }}

  function buildSelect(el, options, value) {{
    el.innerHTML = '';
    for (const opt of options) {{
      const o = document.createElement('option');
      o.value = opt;
      o.textContent = opt;
      el.appendChild(o);
    }}
    if (value !== undefined && value !== null) el.value = value;
  }}

  function buildYearMonthOptions() {{
    const years = uniq(ticketsAll.map(yearOf).filter(Boolean)).sort();
    const yearSel = document.getElementById('fYear');
    const monthSel = document.getElementById('fMonth');

    buildSelect(yearSel, ['全部', ...years], '全部');
    buildSelect(monthSel, ['全部'], '全部');

    function refreshMonths() {{
      const y = yearSel.value;
      let months = [];
      if (y === '全部') {{
        months = uniq(ticketsAll.map(monthOf).filter(Boolean)).sort();
      }} else {{
        months = uniq(ticketsAll.filter(t => yearOf(t) === y).map(monthOf).filter(Boolean)).sort();
      }}
      buildSelect(monthSel, ['全部', ...months.map(m => m)], '全部');
      render();
    }}
    yearSel.addEventListener('change', refreshMonths);
    monthSel.addEventListener('change', render);
    refreshMonths();
  }}

  function filterTickets() {{
    const passenger = document.getElementById('fPassenger').value;
    const y = document.getElementById('fYear').value;
    const m = document.getElementById('fMonth').value;
    const start = document.getElementById('fStart').value; // YYYY-MM-DD
    const end = document.getElementById('fEnd').value;
    const train = document.getElementById('fTrain').value.trim();
    const station = document.getElementById('fStation').value.trim();
    const minP = document.getElementById('fMinPrice').value;
    const maxP = document.getElementById('fMaxPrice').value;
    const seat = document.getElementById('fSeat').value;
    const status = document.getElementById('fStatus').value;

    const minPrice = minP === '' ? null : Number(minP);
    const maxPrice = maxP === '' ? null : Number(maxP);

    return ticketsAll.filter(t => {{
      const d = parseDepartDate(t);
      if (passenger !== '全部' && safe(t.passengerName) !== passenger) return false;
      if (y !== '全部' && yearOf(t) !== y) return false;
      if (m !== '全部' && monthOf(t) !== m) return false;
      if (start && (!d || d < start)) return false;
      if (end && (!d || d > end)) return false;
      if (train && !safe(t.trainNo).toUpperCase().includes(train.toUpperCase())) return false;
      if (station) {{
        const fs = safe(t.fromStation);
        const ts = safe(t.toStation);
        if (!fs.includes(station) && !ts.includes(station)) return false;
      }}
      if (seat !== '全部' && safe(t.seatType) !== seat) return false;
      if (status !== 'all' && safe(t.status) !== status) return false;
      const p = (t.priceYuan === null || t.priceYuan === undefined) ? null : Number(t.priceYuan);
      if (minPrice !== null && (p === null || p < minPrice)) return false;
      if (maxPrice !== null && (p === null || p > maxPrice)) return false;
      return true;
    }});
  }}

  function countBy(ts, keyFn) {{
    const m = new Map();
    for (const t of ts) {{
      const k = keyFn(t);
      if (!k) continue;
      m.set(k, (m.get(k) || 0) + 1);
    }}
    return [...m.entries()].sort((a,b) => b[1]-a[1]);
  }}

  function topBars(el, items, topN=10) {{
    el.innerHTML = '';
    const slice = items.slice(0, topN);
    const max = slice.length ? slice[0][1] : 0;
    for (const [label, val] of slice) {{
      const row = document.createElement('div');
      row.className = 'barrow';
      row.innerHTML = `
        <div class="muted" title="${{label}}">${{label.length>10 ? label.slice(0,10)+'…' : label}}</div>
        <div class="bar"><i style="width:${{max? (val/max*100).toFixed(1):0}}%"></i></div>
        <div style="text-align:right">${{val}}</div>
      `;
      el.appendChild(row);
    }}
    if (!slice.length) {{
      const d = document.createElement('div');
      d.className = 'small';
      d.textContent = '（无数据）';
      el.appendChild(d);
    }}
  }}

  function renderTable(ts) {{
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = '';
    for (const t of ts) {{
      const tr = document.createElement('tr');
      const routeText = `${{safe(t.fromStation)}}→${{safe(t.toStation)}}`;
      const seat = [safe(t.seatType), safe(t.seatRaw)].filter(Boolean).join(' / ');
      const st = safe(t.status) || 'unknown';
      const price = t.priceYuan !== null && t.priceYuan !== undefined ? fmtYuan(Number(t.priceYuan)) : '-';
      tr.innerHTML = `
        <td>${{safe(t.departDateTime).replace('T',' ')}}</td>
        <td><span class="chip">${{safe(t.trainNo)}}</span></td>
        <td><span class="route" title="${{routeText}}">${{routeText}}</span></td>
        <td>${{safe(t.passengerName)}}</td>
        <td>${{seat || '-'}}</td>
        <td>${{price}}</td>
        <td><span class="status ${{st}}">${{statusLabel(st)}} (${{st}})</span></td>
        <td>
          <details>
            <summary>详情</summary>
            <div class="small" style="margin-top:6px">
              <div>订单号：${{safe(t.orderId) || '-'}}</div>
              <div>事件：${{eventLabel(safe(t.eventType))}} (${{safe(t.eventType)}})</div>
              <div>办理日期：${{safe(t.transactionDate) || '-'}}</div>
              <div>邮件日期：${{safe(t.mailDate) || '-'}}</div>
              <div>检票口：${{safe(t.checkInGate) || '-'}}</div>
              <div style="white-space: nowrap;">来源：${{getFileName(safe(t.sourceFile))}} (L${{safe(t.sourceStartLine)}}-L${{safe(t.sourceEndLine)}})</div>
            </div>
          </details>
        </td>
      `;
      tbody.appendChild(tr);
    }}
  }}

  function render() {{
    const ts = filterTickets();

    // KPI
    document.getElementById('kpiTickets').textContent = ts.length;
    const tripKey = (t) => `${{safe(t.trainNo)}}|${{safe(t.departDateTime)}}|${{safe(t.fromStation)}}|${{safe(t.toStation)}}|${{safe(t.passengerName)}}`;
    document.getElementById('kpiTrips').textContent = uniq(ts.map(tripKey)).length;
    const spend = ts.reduce((acc, t) => acc + (t.priceYuan ? Number(t.priceYuan) : 0), 0);
    document.getElementById('kpiSpend').textContent = fmtYuan(spend);
    document.getElementById('kpiNet').textContent = fmtYuan(DATA.summary.netSpendYuan);
    document.getElementById('kpiStations').textContent = uniq(ts.flatMap(t => [t.fromStation, t.toStation]).filter(Boolean)).length;
    document.getElementById('kpiTrains').textContent = uniq(ts.map(t => t.trainNo).filter(Boolean)).length;

    // charts
    topBars(document.getElementById('chartPrefix'), countBy(ts, t => t.trainTypePrefix || 'OTHER'), 12);
    topBars(document.getElementById('chartSeat'), countBy(ts, t => t.seatType || '未知'), 10);
    topBars(document.getElementById('topStations'), countBy(ts, t => t.fromStation), 10); // 发站
    // 合计站：发+到
    const stationAll = [];
    for (const t of ts) {{
      if (t.fromStation) stationAll.push({{k: t.fromStation}});
      if (t.toStation) stationAll.push({{k: t.toStation}});
    }}
    topBars(document.getElementById('topStations'), countBy(stationAll, x => x.k), 10);
    topBars(document.getElementById('topTrains'), countBy(ts, t => t.trainNo), 10);
    topBars(document.getElementById('topSegments'), countBy(ts, t => (t.fromStation && t.toStation) ? `${{t.fromStation}}-${{t.toStation}}` : null), 10);

    // table
    renderTable(ts);
    document.getElementById('tableHint').textContent = `当前展示 ${{ts.length}} 条记录`;
  }}

  function init() {{
    // meta
    const meta = DATA.metadata || {{}};
    document.getElementById('metaLine').textContent =
      `解析器版本 ${{meta.parserVersion || '-'}} · 生成时间 ${{meta.generatedAt || '-'}} · 输入文件 ${{(meta.inputs||[]).length}} 个`;

    // event chips
    const chips = document.getElementById('eventChips');
    const ec = (DATA.summary && DATA.summary.eventCounts) ? DATA.summary.eventCounts : {{}};
    const parts = Object.entries(ec).sort((a,b)=>a[0].localeCompare(b[0]));
    for (const [k,v] of parts) {{
      const span = document.createElement('span');
      span.className = 'chip';
      span.textContent = `${{eventLabel(k)}} (${{k}}): ${{v}}`;
      chips.appendChild(span);
    }}
    const span2 = document.createElement('span');
    span2.className = 'chip';
    span2.textContent = `净支出（全量事件）: ¥${{fmtYuan(DATA.summary.netSpendYuan)}}`;
    chips.appendChild(span2);

    // seat options
    buildSelect(document.getElementById('fSeat'), computeSeatOptions(ticketsAll), '全部');
    buildSelect(document.getElementById('fPassenger'), computePassengerOptions(ticketsAll), '全部');

    // listeners
    for (const id of ['fPassenger','fStart','fEnd','fTrain','fStation','fMinPrice','fMaxPrice','fSeat','fStatus']) {{
      document.getElementById(id).addEventListener('input', render);
      document.getElementById(id).addEventListener('change', render);
    }}
    document.getElementById('btnReset').addEventListener('click', () => {{
      document.getElementById('fStart').value = '';
      document.getElementById('fEnd').value = '';
      document.getElementById('fTrain').value = '';
      document.getElementById('fStation').value = '';
      document.getElementById('fMinPrice').value = '';
      document.getElementById('fMaxPrice').value = '';
      document.getElementById('fSeat').value = '全部';
      document.getElementById('fPassenger').value = '全部';
      document.getElementById('fStatus').value = 'all';
      document.getElementById('fYear').value = '全部';
      // 触发刷新月份
      const ev = new Event('change');
      document.getElementById('fYear').dispatchEvent(ev);
      render();
    }});

    buildYearMonthOptions();
    render();
  }}

  init();
  </script>
</body>
</html>
"""


def write_outputs(
    out_dir: Path,
    events: List[Dict[str, Any]],
    tickets: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    generate_report: bool,
) -> None:
    _ensure_dir(out_dir)

    events_path = out_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    tickets_path = out_dir / "tickets.json"
    tickets_path.write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # 额外导出：按来源文件单独一份（便于分别统计/备份）
    by_file_dir = out_dir / "by_file"
    _ensure_dir(by_file_dir)

    # 分组
    events_by: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        src = ev.get("sourceFile") or "unknown"
        events_by.setdefault(src, []).append(ev)
    tickets_by: Dict[str, List[Dict[str, Any]]] = {}
    for t in tickets:
        src = t.get("sourceFile") or "unknown"
        tickets_by.setdefault(src, []).append(t)

    # 写出
    for src_file in sorted(set(events_by.keys()) | set(tickets_by.keys())):
        src_path = Path(src_file)
        stem = src_path.stem if src_path.stem else "unknown"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
        sub = by_file_dir / safe_stem
        _ensure_dir(sub)

        evs = events_by.get(src_file, [])
        tks = tickets_by.get(src_file, [])

        with (sub / "events.jsonl").open("w", encoding="utf-8") as f:
            for ev in evs:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        (sub / "tickets.json").write_text(json.dumps(tks, ensure_ascii=False, indent=2), encoding="utf-8")
        (sub / "metadata.json").write_text(
            json.dumps(
                {
                    "parserVersion": metadata.get("parserVersion"),
                    "generatedAt": metadata.get("generatedAt"),
                    "sourceFile": src_file,
                    "counts": {"events": len(evs), "tickets": len(tks)},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # 分文件 report.html（不合并版本）
        if generate_report:
            (sub / "report.html").write_text(
                build_report_html(
                    tks,
                    evs,
                    {
                        "parserVersion": metadata.get("parserVersion"),
                        "generatedAt": metadata.get("generatedAt"),
                        "inputs": [{"path": src_file}],
                        "counts": {"events": len(evs), "tickets": len(tks)},
                    },
                ),
                encoding="utf-8",
            )

    if generate_report:
        report_path = out_dir / "report.html"
        report_path.write_text(build_report_html(tickets, events, metadata), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="解析 12306 txt 通知邮件为 JSON，并生成离线报表。")
    ap.add_argument("--input-dir", type=str, default=None, help="输入目录（扫描 *.txt）")
    ap.add_argument("--inputs", nargs="*", default=None, help="显式指定输入 txt 文件列表")
    ap.add_argument("--output", type=str, default="out", help="输出目录")
    ap.add_argument("--no-report", action="store_true", help="不生成 report.html")
    ap.add_argument("--hash-inputs", action="store_true", help="metadata.json 里写入输入文件 sha256（更可复现，但稍慢）")
    args = ap.parse_args(argv)

    input_files: List[Path] = []
    if args.inputs:
        input_files = [Path(p).expanduser().resolve() for p in args.inputs]
    else:
        in_dir = Path(args.input_dir or "data").expanduser().resolve()
        if not in_dir.exists() or not in_dir.is_dir():
            print(f"输入目录不存在：{in_dir}", file=sys.stderr)
            return 2
        input_files = sorted(in_dir.glob("*.txt"), key=lambda p: p.name)

    if not input_files:
        print("未找到输入文件。请使用 --input-dir 或 --inputs。", file=sys.stderr)
        return 2

    # 记录输入摘要，确保可复现
    inputs_meta: List[Dict[str, Any]] = []
    for p in input_files:
        try:
            stat = p.stat()
            item = {
                "path": str(p),
                "name": p.name,
                "sizeBytes": stat.st_size,
                "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
            if args.hash_inputs:
                item["sha256"] = _sha256_file(p)
            inputs_meta.append(item)
        except Exception as e:
            inputs_meta.append({"path": str(p), "error": str(e)})

    events, tickets = parse_events_from_files(input_files)
    label_ticket_status(events, tickets)

    metadata = {
        "parserVersion": PARSER_VERSION,
        "generatedAt": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs_meta,
        "counts": {
            "events": len(events),
            "tickets": len(tickets),
        },
    }

    out_dir = Path(args.output).expanduser().resolve()
    write_outputs(out_dir, events, tickets, metadata, generate_report=(not args.no_report))

    print(f"输出目录：{out_dir}")
    print(f"- events.jsonl: {out_dir / 'events.jsonl'}")
    print(f"- tickets.json: {out_dir / 'tickets.json'}")
    print(f"- metadata.json: {out_dir / 'metadata.json'}")
    if not args.no_report:
        print(f"- report.html: {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


