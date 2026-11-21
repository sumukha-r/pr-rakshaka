def total_price(order_ids, db):
    total = 0
    for oid in order_ids:
        total += db.query(oid).price
    return total
