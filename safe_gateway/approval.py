from .models import ApprovalDecision, ApprovalItem, UserRole


class ApprovalQueue:
    def __init__(self):
        self._items: dict[str, ApprovalItem] = {}

    def add(self, item: ApprovalItem) -> ApprovalItem:
        self._items[item.id] = item
        return item

    def list_pending(self) -> list[ApprovalItem]:
        return [item for item in self._items.values() if item.status == ApprovalDecision.pending]

    def list_all(self) -> list[ApprovalItem]:
        return list(self._items.values())

    def decide(self, approval_id: str, approved: bool, reviewer_role: UserRole) -> ApprovalItem:
        if approval_id not in self._items:
            raise KeyError(f"Approval item {approval_id} was not found.")
        item = self._items[approval_id]
        item.status = ApprovalDecision.approved if approved else ApprovalDecision.rejected
        item.reviewer_role = reviewer_role
        self._items[approval_id] = item
        return item

