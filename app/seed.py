"""Seed ORBIT with demo business data and a small knowledge base.

Run with:  python -m app.seed

The data is a small support desk: customers, their orders, and their complaint
history, plus the policies that govern replacements, refunds and escalation.
That combination is what makes the flagship demonstration work — the agents
have to retrieve a policy, check a customer's record against it, and only then
propose an action a human approves.
"""

from __future__ import annotations

from app.memory.store import get_store
from app.memory.vector import get_memory

CUSTOMERS = [
    ("CUST-001", "Ananya Sharma", "ananya.sharma@example.com", "gold", "2023-02-14"),
    ("CUST-002", "Rohit Verma", "rohit.verma@example.com", "silver", "2024-06-02"),
    ("CUST-003", "Meera Nair", "meera.nair@example.com", "gold", "2022-11-30"),
    ("CUST-004", "Imran Qureshi", "imran.qureshi@example.com", "bronze", "2025-01-19"),
    ("CUST-005", "Divya Menon", "divya.menon@example.com", "silver", "2024-09-08"),
]

ORDERS = [
    ("ORD-1001", "CUST-001", "Aurora Wireless Headphones", 8499.00, "delivered", "2026-07-02"),
    ("ORD-1002", "CUST-001", "Aurora Charging Dock", 1999.00, "delivered", "2026-08-11"),
    ("ORD-1003", "CUST-002", "Nimbus Mechanical Keyboard", 6499.00, "delivered", "2026-08-21"),
    ("ORD-1004", "CUST-003", "Vega 27\" 4K Monitor", 32999.00, "delivered", "2026-06-15"),
    ("ORD-1005", "CUST-003", "Vega Monitor Arm", 4299.00, "delivered", "2026-06-15"),
    ("ORD-1006", "CUST-004", "Pulse Fitness Band", 3499.00, "delivered", "2026-09-01"),
    ("ORD-1007", "CUST-005", "Nimbus Ergonomic Mouse", 2799.00, "shipped", "2026-09-09"),
]

COMPLAINTS = [
    ("CMP-5001", "CUST-001", "ORD-1001", "Right earcup went silent",
     "The right earcup stopped producing sound about three weeks after delivery. "
     "A factory reset and firmware update were attempted and neither restored audio.",
     "open", "2026-08-28"),
    ("CMP-5002", "CUST-001", "ORD-1001", "Charging case not holding charge",
     "Reported earlier the same month: the charging case discharges overnight even when unused. "
     "Support advised a firmware update, which did not resolve it.",
     "resolved", "2026-08-05"),
    ("CMP-5003", "CUST-002", "ORD-1003", "Three keys unresponsive",
     "The J, K and L keys register intermittently. The issue appeared within the warranty window.",
     "open", "2026-09-05"),
    ("CMP-5004", "CUST-003", "ORD-1004", "Dead pixel cluster",
     "A cluster of roughly nine dead pixels is visible in the lower-left quadrant. "
     "This is the customer's second monitor issue this year.",
     "open", "2026-09-10"),
    ("CMP-5005", "CUST-004", "ORD-1006", "Strap clasp broke",
     "The clasp snapped after two weeks of normal use. Customer has requested a replacement strap.",
     "open", "2026-09-14"),
]

KNOWLEDGE = [
    {
        "kind": "policy",
        "source": "policy/replacement-v3",
        "text": (
            "Replacement Policy (v3, effective April 2026). A customer is eligible for a free "
            "product replacement when all of the following hold: the order was delivered within "
            "the last 180 days; the fault is a hardware defect rather than accidental damage; and "
            "at least one documented troubleshooting attempt has failed. Customers on the gold "
            "tier are additionally eligible for advance replacement, meaning the replacement ships "
            "before the faulty unit is returned. A second complaint about the same order within "
            "60 days escalates the case to a senior reviewer regardless of tier."
        ),
    },
    {
        "kind": "policy",
        "source": "policy/refund-v2",
        "text": (
            "Refund Policy (v2). Refunds are issued only when a replacement is unavailable or has "
            "already failed once. Full refunds apply within 30 days of delivery. Between 31 and "
            "180 days a prorated refund applies, reduced by 10 percent for every complete 30 days "
            "since delivery. No refund is issued after 180 days. Any refund above 5,000 rupees "
            "requires human approval before it is released."
        ),
    },
    {
        "kind": "policy",
        "source": "policy/escalation-v1",
        "text": (
            "Escalation Policy (v1). A support case escalates to a human reviewer when the "
            "automated confidence in the recommendation falls below 65 percent, when the action "
            "involves money leaving the business, when a message will be sent to an external "
            "recipient, or when the customer has filed three or more complaints in 90 days. "
            "Escalated cases must record the reviewer's identity and the reason for the decision."
        ),
    },
    {
        "kind": "policy",
        "source": "policy/warranty-v2",
        "text": (
            "Warranty terms. Audio products carry a 12-month manufacturer warranty covering "
            "hardware defects. Input devices such as keyboards and mice carry 24 months. Displays "
            "carry 36 months, and a dead-pixel cluster of five or more adjacent pixels qualifies "
            "as a covered defect. Accessories including straps, cases and docks carry 6 months. "
            "Warranty does not cover liquid ingress, physical damage or unauthorised repair."
        ),
    },
    {
        "kind": "document",
        "source": "handbook/support-playbook",
        "text": (
            "Support playbook. Open every case by confirming the customer's identity and order "
            "history before discussing remedies. Check whether the same order has generated a "
            "prior complaint, since repeat faults change the remedy. Offer the least disruptive "
            "remedy that resolves the issue: troubleshooting first, then replacement, then refund. "
            "Never promise a timeline the fulfilment system has not confirmed."
        ),
    },
    {
        "kind": "document",
        "source": "handbook/product-notes",
        "text": (
            "Known product issues. Aurora Wireless Headphones manufactured in batch A-2026-03 have "
            "a documented fault in the right-channel driver, presenting as sudden loss of audio in "
            "the right earcup. Affected units qualify for replacement without further diagnosis. "
            "Nimbus Mechanical Keyboards shipped before July 2026 may show intermittent key "
            "registration caused by a loose ribbon connector, which is a covered hardware defect."
        ),
    },
    {
        "kind": "document",
        "source": "handbook/data-handling",
        "text": (
            "Data handling. Agents may read customer records needed for the case in front of them "
            "and nothing wider. Customer email addresses must not be included in any outbound "
            "message other than one addressed to that customer. Any action that writes to a "
            "customer record is logged with the task id, the agent that proposed it and the "
            "reviewer who approved it."
        ),
    },
]


def seed_business_data() -> dict[str, int]:
    store = get_store()

    for row in CUSTOMERS:
        if not store.query("SELECT id FROM customers WHERE id = ?", (row[0],)):
            store.execute(
                "INSERT INTO customers (id, name, email, tier, joined_at) VALUES (?,?,?,?,?)", row
            )
    for row in ORDERS:
        if not store.query("SELECT id FROM orders WHERE id = ?", (row[0],)):
            store.execute(
                "INSERT INTO orders (id, customer_id, product, amount, status, ordered_at) "
                "VALUES (?,?,?,?,?,?)", row
            )
    for row in COMPLAINTS:
        if not store.query("SELECT id FROM complaints WHERE id = ?", (row[0],)):
            store.execute(
                "INSERT INTO complaints (id, customer_id, order_id, subject, detail, status, created_at) "
                "VALUES (?,?,?,?,?,?,?)", row
            )
    return {"customers": len(CUSTOMERS), "orders": len(ORDERS), "complaints": len(COMPLAINTS)}


def seed_knowledge() -> int:
    memory = get_memory()
    added = 0
    for item in KNOWLEDGE:
        memory.add(item["text"], source=item["source"], kind=item["kind"])
        added += 1

    # Index the complaint history too, so semantic recall of a past issue works
    # even when nobody knows the complaint id to look up.
    for complaint_id, customer_id, order_id, subject, detail, status, created in COMPLAINTS:
        memory.add(
            f"Complaint {complaint_id} ({status}) from {customer_id} about order {order_id}: "
            f"{subject}. {detail}",
            source=f"complaint/{complaint_id}",
            kind="record",
            metadata={"customer_id": customer_id, "order_id": order_id},
        )
        added += 1
    return added


def seed_all(verbose: bool = True) -> dict[str, int]:
    counts = seed_business_data()
    counts["knowledge_chunks"] = seed_knowledge()
    if verbose:
        store = get_store()
        memory = get_memory()
        print("ORBIT seed complete")
        print(f"  structured store : {store.backend}")
        print(f"  semantic memory  : {memory.backend} ({memory.count()} records)")
        for key, value in counts.items():
            print(f"  {key:<17}: {value}")
    return counts


if __name__ == "__main__":
    seed_all()
