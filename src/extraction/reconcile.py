def reconcile(unique_rows, invoice_total, tolerance_pct=0.005, tolerance_abs=1.0):
    """
    unique_rows: list of deduped line-item dicts
    invoice_total: parsed total from OCR
    """
    computed_total = sum(
        r.get("item_amount") or 0 for r in unique_rows
    )

    if invoice_total is None:
        return {
            "computed_total": computed_total,
            "invoice_total": None,
            "difference": None,
            "ok": True,
            "suggestions": ["No total row found"]
        }

    diff = invoice_total - computed_total

    ok = True
    suggestions = []

    # check absolute tolerance
    if abs(diff) > tolerance_abs:
        ok = False

    # check percentage tolerance
    if invoice_total > 0:
        pct = abs(diff) / invoice_total
        if pct > tolerance_pct:
            ok = False

    if not ok:
        # Look for misclassified rows
        suggestions.append("Possible missing items or misclassified totals.")

        # Identify rows that were not classified as line items but contain numbers
        # (requires you to pass raw_rows)
        # suggestions.append(...)

    return {
        "computed_total": computed_total,
        "invoice_total": invoice_total,
        "difference": diff,
        "ok": ok,
        "suggestions": suggestions
    }
