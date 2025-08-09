# We had to report in to the LDP leader whether we had successfully brushed our teeth with our offhand each day / night.
# I've generally done well, but the group was then required to report as one.
# I explored using this script to scrape iMessage for the results of the group text and send it in.
# However, the leader confirmed that we were not allowed to use scripts to automate the reporting.

from pathlib import Path
from datetime import datetime, timezone, timedelta
from imessage_reader import fetch_data

DB_PATH = str(Path.home() / "Library" / "Messages" / "chat.db")
fd = fetch_data.FetchData(DB_PATH)

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
def apple_to_dt(t):
    # Handles ints/floats in Apple epoch seconds; pass through if it's already a datetime
    if isinstance(t, (int, float)):
        return APPLE_EPOCH + timedelta(seconds=t)
    return t

def get_service(m):
    # Works whether rows are dicts or tuples from different package versions
    try:
        return (m.get("service") or "").lower()
    except AttributeError:
        try:
            return str(m[3]).lower()   # common tuple layout: (..., text, timestamp, service, ...)
        except Exception:
            return ""

def get_ts(m):
    try:
        return m.get("date") or m.get("timestamp")
    except AttributeError:
        return m[2]  # common tuple index for timestamp

def get_sender(m):
    try:
        # Prefer explicit fields if present
        return m.get("handle") or m.get("phone_number") or m.get("sender") or "unknown"
    except AttributeError:
        # For a common tuple layout: (handle, text, ts, service, account, is_from_me)
        return m[0] if m and len(m) > 0 else "unknown"

def get_text(m):
    try:
        return m.get("text") or ""
    except AttributeError:
        return m[1] if m and len(m) > 1 else ""

def from_me(m):
    try:
        return bool(m.get("is_from_me"))
    except AttributeError:
        return bool(m[5]) if m and len(m) > 5 else False

rows = fd.get_messages()

# Keep only SMS, newest last
sms_rows = [m for m in rows if get_service(m) == "sms"]
latest_10 = sorted(sms_rows, key=lambda m: get_ts(m))[-10:]

for m in latest_10:
    ts = apple_to_dt(get_ts(m))
    who = "me" if from_me(m) else get_sender(m)
    print(f"[{ts}] (SMS) {who}: {get_text(m)}")
