# -*- coding: utf-8 -*-
"""Mock tool environments + single-call and multi-step tool-calling tasks (lfm3-tools PUBLIC task set).

Stdlib only; runs on Python 3.9 and 3.12.  Tasks are graded on the FINAL STATE
of the mock environment (task["check"](env.state)).

Layout
  ENVS   name -> environment class (TOOLS in OpenAI chat-completions format,
         __init__(init), call(name, args) which never raises and leaves the
         state unchanged whenever it returns {"error": ...})
  TASKS  S01..S30  single-call family, 6 per environment: the prompt holds
                   everything needed and exactly ONE write call solves it (read
                   calls are allowed but unnecessary; the reference is that call)
         M01..M30  multi-step family, 6 per environment: references of 6-14
                   calls, rules live in policy read-tools that the write tools
                   enforce, at least one re-read of state a write has changed,
                   and in many tasks a step the policy refuses, so that a
                   different route is needed
         C1..C8    calibration (C1, C3, C5, C7 single-call; C2, C4, C6, C8
                   multi-step; every environment has at least one), same style
                   and difficulty, used only to check difficulty with a model
                   outside the comparison
         Each task dict: id, family ("single" | "multi"), calibration (bool),
         env, prompt, init, reference, ref_errors, trap, check, features,
         mentions, creates, seq.

Domains (all new relative to the bonsai2 and lfm25-tools rounds):
  vet        veterinary clinic: appointments, follow-ups, vaccination records,
             weighings and weight-based dosing lookups
  warehouse  warehouse pick/pack: orders, bin locations, stock lots (FEFO),
             carrier cut-offs and parcel shipments
  lab        research-lab sample custody: samples, freezers/racks,
             chain-of-custody transfers, retention and destruction rules
  league     community sports league: fixtures, venues, referee assignments,
             team sheets with eligibility and suspension rules
  museum     museum collections: object loans, condition reports, gallery
             moves and light-exposure limits
  Nothing involves banks, payments, money, prices or invoices.

  * no schema uses a union type and no task needs null / "" as an argument
  * the same lenient argument coercions as lfm25-tools ("250" -> 250, "true" -> true,
    optional argument sent as null == omitted), plus an array sent as a JSON string
  * nested arguments are one level deep: arrays of scalars, or arrays of flat
    objects (e.g. a list of {vaccine, batch_number, route})
  * every entity a prompt names is mechanically checked to exist in init
  * "today" comes from a read tool (or, in single-call prompts, is stated in
    the prompt), never from the real clock
  * single-call checks compare the whole final state, so any extra or
    different write fails; the verifier also perturbs every scalar argument of
    each single-call reference and checks that every changed outcome fails

Run this file directly to self-verify:  prints "OK 68/68".
"""
import copy
import datetime
import itertools
import json
import random
import re


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------
class _Rule(Exception):
    """A business-rule / lookup failure; turned into {"error": ...}."""


def _norm(schema, v, path):
    """Validate v against a (small subset of) JSON Schema. Returns (value, err)."""
    t = schema.get("type")
    if v is None:
        return None, "'%s' must not be null" % path
    if t == "string":
        if not isinstance(v, str):
            return None, "'%s' must be a string" % path
        if "enum" in schema and v not in schema["enum"]:
            return None, "'%s' must be one of: %s" % (path, ", ".join(schema["enum"]))
        return v, None
    if t == "integer":
        if isinstance(v, bool):
            return None, "'%s' must be an integer" % path
        if isinstance(v, str) and re.match(r"^-?\d{1,9}$", v.strip()):
            v = int(v.strip())   # "250" for 250 is a tool-call formatting slip, not a reasoning error
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")) or v != int(v):
                return None, "'%s' must be an integer" % path
            v = int(v)
        if not isinstance(v, int):
            return None, "'%s' must be an integer" % path
        if abs(v) > 10 ** 9:
            return None, "'%s' is out of range" % path
        if "minimum" in schema and v < schema["minimum"]:
            return None, "'%s' must be >= %d" % (path, schema["minimum"])
        if "maximum" in schema and v > schema["maximum"]:
            return None, "'%s' must be <= %d" % (path, schema["maximum"])
        return v, None
    if t == "boolean":
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"   # same leniency for "true"/"false" sent as strings
        if not isinstance(v, bool):
            return None, "'%s' must be true or false" % path
        return v, None
    if t == "array":
        if isinstance(v, str) and v.strip().startswith("["):
            try:                              # an array sent as a JSON string is a formatting slip too
                v = json.loads(v)
            except ValueError:
                pass
        if not isinstance(v, list):
            return None, "'%s' must be an array" % path
        if "minItems" in schema and len(v) < schema["minItems"]:
            return None, "'%s' needs at least %d item(s)" % (path, schema["minItems"])
        if "maxItems" in schema and len(v) > schema["maxItems"]:
            return None, "'%s' takes at most %d item(s)" % (path, schema["maxItems"])
        out = []
        for n, item in enumerate(v):
            val, err = _norm(schema["items"], item, "%s[%d]" % (path, n))
            if err:
                return None, err
            out.append(val)
        return out, None
    if t == "object":
        if not isinstance(v, dict):
            return None, "'%s' must be an object" % path
        props = schema.get("properties", {})
        required = schema.get("required", [])
        pre = "" if path == "arguments" else path + "."
        out = {}
        for k in v:
            if not isinstance(k, str) or k not in props:
                return None, "unexpected argument '%s%s' (allowed: %s)" % (
                    pre, str(k)[:40], ", ".join(sorted(props)) or "none")
        for k in required:
            if k not in v:
                return None, "missing required argument '%s%s'" % (pre, k)
        for k, val in v.items():
            if val is None and k not in required:
                continue  # optional argument sent as null == omitted
            val, err = _norm(props[k], val, pre + k)
            if err:
                return None, err
            out[k] = val
        return out, None
    return None, "'%s' has an unsupported type" % path


def _fn(name, desc, props=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {},
                       "required": list(required), "additionalProperties": False}}}


def _s(desc, enum=None):
    d = {"type": "string", "description": desc}
    if enum:
        d["enum"] = list(enum)
    return d


def _i(desc, minimum=None, maximum=None):
    d = {"type": "integer", "description": desc}
    if minimum is not None:
        d["minimum"] = minimum
    if maximum is not None:
        d["maximum"] = maximum
    return d


def _b(desc):
    return {"type": "boolean", "description": desc}


def _a(desc, items, min_items=None, max_items=None):
    d = {"type": "array", "description": desc, "items": items}
    if min_items is not None:
        d["minItems"] = min_items
    if max_items is not None:
        d["maxItems"] = max_items
    return d


def _o(props, required=()):
    """A flat object (used as the item type of an array argument)."""
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_PAGE = _i("Page number, starting at 1 (default 1).", 1)


def _pdate(s, what="date"):
    if not isinstance(s, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        raise _Rule("%s must be formatted YYYY-MM-DD, got '%s'" % (what, str(s)[:30]))
    try:
        return datetime.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
    except ValueError:
        raise _Rule("%s '%s' is not a real calendar date" % (what, s))


def _ptime(s, what="time"):
    if not isinstance(s, str) or not re.match(r"^\d{2}:\d{2}$", s):
        raise _Rule("%s must be formatted HH:MM (24h), got '%s'" % (what, str(s)[:30]))
    h, m = int(s[0:2]), int(s[3:5])
    if h > 23 or m > 59:
        raise _Rule("%s '%s' is not a valid time" % (what, s))
    return h * 60 + m


def _fmt_time(minutes):
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def _plus_days(s, n):
    return (_pdate(s) + datetime.timedelta(days=n)).isoformat()


def _days_between(a, b):
    return (_pdate(b) - _pdate(a)).days


def _paged(rows, page, size):
    """Returns (meta, slice). Raises _Rule for a page that does not exist."""
    pages = max(1, (len(rows) + size - 1) // size)
    if page > pages:
        raise _Rule("page %d does not exist (there are %d page(s))" % (page, pages))
    return ({"page": page, "pages": pages, "total_results": len(rows)},
            rows[(page - 1) * size: page * size])


def _text(v, what, maxlen=80):
    v = v.strip()
    if not v:
        raise _Rule("%s must not be empty" % what)
    if len(v) > maxlen:
        raise _Rule("%s is too long (at most %d characters)" % (what, maxlen))
    return v


_DAYNAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DAY3 = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_READ_PREFIXES = ("get_", "list_", "find_", "search_")


def _is_read(name):
    return isinstance(name, str) and name.startswith(_READ_PREFIXES)


def _clone(state):
    """Deep copy of a JSON-shaped state (much faster than copy.deepcopy)."""
    return json.loads(json.dumps(state))


class _Env(object):
    TOOLS = []

    def __init__(self, init):
        self.state = _clone(init)

    def _restore(self, snap):
        self.state.clear()
        self.state.update(snap)

    def call(self, name, args):
        try:
            return self._call(name, args)
        except Exception as e:  # never raise
            return {"error": "invalid call (%s)" % type(e).__name__}

    def _call(self, name, args):
        if not isinstance(name, str):
            return {"error": "tool name must be a string"}
        spec = None
        for t in self.TOOLS:
            if t["function"]["name"] == name:
                spec = t["function"]
        if spec is None:
            return {"error": "unknown tool '%s'; available tools: %s" % (
                name[:60], ", ".join(t["function"]["name"] for t in self.TOOLS))}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        args, err = _norm(spec["parameters"], args, "arguments")
        if err:
            return {"error": err}
        snap = _clone(self.state)
        try:
            res = getattr(self, "_t_" + name)(args)
            res = json.loads(json.dumps(res))
            json.dumps(self.state)
        except _Rule as e:
            self._restore(snap)
            return {"error": str(e)}
        except Exception as e:
            self._restore(snap)
            return {"error": "internal error (%s); nothing was changed" % type(e).__name__}
        if isinstance(res, dict) and "error" in res:
            self._restore(snap)
        return res

    def _get(self, coll, key, what):
        rec = self.state[coll].get(key)
        if rec is None:
            raise _Rule("unknown %s '%s'" % (what, key[:40]))
        return rec

    def _new_id(self, counter, prefix):
        n = self.state[counter]
        self.state[counter] = n + 1
        return "%s-%d" % (prefix, n)

    def _today(self):
        return _pdate(self.state["today"])


# --------------------------------------------------------------------------
# check helpers + task registry
# --------------------------------------------------------------------------
def _fits(rec, want):
    """A created record matches a wanted one only if it has exactly the wanted fields."""
    if set(rec.keys()) != set(want.keys()):
        return False
    for k, v in want.items():
        if callable(v):
            if not v(rec.get(k)):
                return False
        elif rec.get(k) != v:
            return False
    return True


def _match(fresh, wants):
    if len(fresh) != len(wants):
        return False
    for perm in itertools.permutations(range(len(fresh))):
        if all(_fits(fresh[perm[n]], wants[n]) for n in range(len(wants))):
            return True
    return False


def _is_collection(v):
    return isinstance(v, dict) and len(v) > 0 and all(isinstance(x, dict) for x in v.values())


def _final(s, i, changes=None, new=None):
    """True iff state s equals init i except for exactly the described differences.

    changes: {collection: {record_id: {field: value | predicate}}}; for non-collection top-level keys
             {key: value | predicate}.
    new:     {collection: [ {field: value | predicate}, ... ]} records that must have been created
             (matched by content, generated ids ignored; every field must be listed). Collections not
             listed must have no new records.
    Counters (next_*) are ignored. Every record and field not mentioned must be identical to init.
    """
    changes = changes or {}
    new = new or {}
    if not isinstance(s, dict) or set(s.keys()) != set(i.keys()):
        return False
    for key in i:
        if key.startswith("next_"):
            continue
        if _is_collection(i[key]):
            if not isinstance(s[key], dict):
                return False
            for rid, rec in i[key].items():
                got = s[key].get(rid)
                if not isinstance(got, dict) or set(got.keys()) != set(rec.keys()):
                    return False
                ch = changes.get(key, {}).get(rid, {})
                for f in rec:
                    want = ch[f] if f in ch else rec[f]
                    if callable(want):
                        if not want(got[f]):
                            return False
                    elif got[f] != want:
                        return False
            fresh = [s[key][k] for k in sorted(s[key]) if k not in i[key]]
            if not _match(fresh, new.get(key, [])):
                return False
        elif key in changes:
            want = changes[key]
            if callable(want):
                if not want(s[key]):
                    return False
            elif s[key] != want:
                return False
        elif s[key] != i[key]:
            return False
    return True


def _any(*options):
    """Pass if any of several complete (changes, new) outcomes holds."""
    return lambda s, i: any(_final(s, i, changes=c, new=n) for c, n in options)


def _set(xs):
    xs = sorted(xs)
    return lambda v: isinstance(v, list) and sorted(v) == xs


def _anystr(v):
    """Free-text optional fields (notes, reasons) are not graded."""
    return isinstance(v, str)


TASKS = []


def _add(tid, env, init, prompt, reference, trap, check, features, ref_errors=(), mentions=(), creates=(), seq=()):
    """mentions: strings the prompt names that must exist in init (checked mechanically, with the ids/quoted
    names the verifier extracts itself). creates: prompt strings that are to be created, exempt from that check.
    seq: groups of reference indices whose relative order is part of the task (not permuted by the order test)."""
    frozen = copy.deepcopy(init)
    family = "single" if tid.startswith("S") or (tid.startswith("C") and int(tid[1:]) % 2 == 1) else "multi"
    TASKS.append({
        "id": tid, "family": family, "calibration": tid.startswith("C"), "env": env, "init": init,
        "prompt": prompt, "reference": reference, "ref_errors": list(ref_errors), "trap": trap,
        "features": features, "mentions": list(mentions), "creates": list(creates), "seq": [list(g) for g in seq],
        "check": (lambda state, _c=check, _i=frozen: bool(_c(state, _i))),
    })


# --------------------------------------------------------------------------
# 1. veterinary clinic: patients, appointments, vaccinations, weighings, prescriptions
# --------------------------------------------------------------------------
_SPECIES = ["dog", "cat", "rabbit", "ferret", "bird"]
_APPT_TYPES = ["consult", "vaccination", "dental", "surgery", "nail_trim"]
_VACCINES = ["dhpp", "lepto", "kennel_cough", "rabies", "fvrcp", "felv", "rhd", "myxo"]
_ROUTES = ["subcutaneous", "intramuscular", "intranasal", "oral"]
_DRUGS = ["amoxicillin", "cefalexin", "carprofen", "gabapentin", "maropitant", "enrofloxacin", "prednisolone"]
_CANCEL_REASONS = ["owner_request", "clinic_closure", "patient_deceased", "duplicate_booking", "vet_unavailable"]
_STOP_REASONS = ["course_complete", "adverse_reaction", "dose_change", "owner_request", "patient_deceased"]


class VetEnv(_Env):
    TOOLS = [
        _fn("get_clinic_policy", "Today's date, opening hours, appointment durations, the vets (species, appointment "
            "types and working days), the vaccine schedule, the dosing formulary and the clinic rules that the write "
            "tools enforce."),
        _fn("find_patients", "Search patients by pet name or owner name (case-insensitive substring).",
            {"query": _s("Text that must appear in the pet's or the owner's name."),
             "species": _s("Only this species.", _SPECIES)}, ["query"]),
        _fn("get_patient", "One patient: species, breed, date of birth, owner, status, allergies, weight history, "
            "vaccinations, prescriptions and booked appointments.",
            {"patient_id": _s("Patient id, e.g. A-101.")}, ["patient_id"]),
        _fn("list_appointments", "Appointments, filtered, ordered by date and time; 8 per page.",
            {"date": _s("Only this date, YYYY-MM-DD."), "vet_id": _s("Only this vet, e.g. V-01."),
             "patient_id": _s("Only this patient."),
             "status": _s("Only this status.", ["booked", "completed", "cancelled", "no_show"]), "page": _PAGE}),
        _fn("get_dosing", "Weight-based dose lookup in the clinic formulary: the single dose in mg for a drug, "
            "species and body weight, and the maximum number of doses per day.",
            {"drug": _s("Drug.", _DRUGS), "species": _s("Species.", _SPECIES),
             "weight_grams": _i("Body weight in grams.", 20, 150000)}, ["drug", "species", "weight_grams"]),
        _fn("schedule_appointment", "Book a new appointment. Follow-up visits of an earlier appointment are booked "
            "with book_followup instead.",
            {"patient_id": _s("Patient id."), "vet_id": _s("Vet id."), "date": _s("Date, YYYY-MM-DD."),
             "time": _s("Start time HH:MM (24h, on a 5-minute mark)."), "type": _s("Appointment type.", _APPT_TYPES),
             "reason": _s("Optional short note for the vet."),
             "reminders": _a("Optional reminders to the owner before the visit.",
                             _o({"channel": _s("How to remind.", ["sms", "email", "phone"]),
                                 "days_before": _i("Days before the appointment.", 1, 14)},
                                ["channel", "days_before"]), 1, 3)},
            ["patient_id", "vet_id", "date", "time", "type"]),
        _fn("reschedule_appointment", "Move a booked appointment to another date and time, optionally to another vet. "
            "Patient and type stay the same.",
            {"appointment_id": _s("Appointment id, e.g. AP-3010."), "date": _s("New date, YYYY-MM-DD."),
             "time": _s("New start time HH:MM."), "vet_id": _s("New vet (default: the same vet).")},
            ["appointment_id", "date", "time"]),
        _fn("book_followup", "Book a 15-minute follow-up visit for an earlier, completed appointment, with the vet "
            "who saw the patient then.",
            {"appointment_id": _s("The completed appointment this follows up."), "date": _s("Date, YYYY-MM-DD."),
             "time": _s("Start time HH:MM."), "reason": _s("Optional short note for the vet.")},
            ["appointment_id", "date", "time"]),
        _fn("cancel_appointment", "Cancel a booked appointment.",
            {"appointment_id": _s("Appointment id."), "reason": _s("Why it is cancelled.", _CANCEL_REASONS)},
            ["appointment_id", "reason"]),
        _fn("record_vaccinations", "Record one or more vaccines given to one patient on one date by one vet. The next "
            "due date of each vaccine is set from the vaccine schedule.",
            {"patient_id": _s("Patient id."), "date_given": _s("Date given, YYYY-MM-DD."), "vet_id": _s("Vet id."),
             "vaccines": _a("The vaccines given.",
                            _o({"vaccine": _s("Vaccine.", _VACCINES), "batch_number": _s("Batch number on the vial."),
                                "route": _s("How it was given (default subcutaneous).", _ROUTES)},
                               ["vaccine", "batch_number"]), 1, 6)},
            ["patient_id", "date_given", "vet_id", "vaccines"]),
        _fn("record_weight", "Record a weighing in grams.",
            {"patient_id": _s("Patient id."), "date": _s("Date weighed, YYYY-MM-DD."),
             "weight_grams": _i("Weight in grams.", 20, 150000),
             "body_condition": _i("Optional body condition score on the 9-point scale.", 1, 9)},
            ["patient_id", "date", "weight_grams"]),
        _fn("prescribe_medication", "Start a prescription for a patient (starts today).",
            {"patient_id": _s("Patient id."), "drug": _s("Drug.", _DRUGS), "dose_mg": _i("Single dose in mg.", 1, 5000),
             "times_per_day": _i("Doses per day.", 1, 4), "days": _i("Length of the course in days.", 1, 90),
             "vet_id": _s("Prescribing vet."), "with_food": _b("Give with food (default false).")},
            ["patient_id", "drug", "dose_mg", "times_per_day", "days", "vet_id"]),
        _fn("stop_prescription", "Stop an active prescription.",
            {"prescription_id": _s("Prescription id, e.g. RX-901."), "reason": _s("Why it is stopped.", _STOP_REASONS)},
            ["prescription_id", "reason"]),
        _fn("update_patient", "Change patient details: microchip number, neutered flag, the allergy list (replaces the "
            "whole list) or the record status.",
            {"patient_id": _s("Patient id."), "microchip": _s("15-digit microchip number."),
             "neutered": _b("Neutered."), "allergies": _a("Complete list of drugs the patient is allergic to.",
                                                          _s("Drug.", _DRUGS)),
             "status": _s("Record status.", ["active", "deceased", "transferred"])},
            ["patient_id"]),
    ]

    def _patient(self, pid, active=True):
        p = self._get("patients", pid, "patient")
        if active and p["status"] != "active":
            raise _Rule("patient %s (%s) is %s" % (pid, p["name"], p["status"]))
        return p

    def _vet(self, vid):
        return self._get("vets", vid, "vet")

    def _dur(self, typ):
        return self.state["policy"]["durations"][typ]

    def _latest_weight(self, pid):
        rows = sorted((w["date"], k) for k, w in self.state["weights"].items() if w["patient"] == pid)
        return (rows[-1][1], self.state["weights"][rows[-1][1]]) if rows else (None, None)

    def _dose(self, drug, grams):
        f = self.state["policy"]["formulary"][drug]
        return min(f["max_mg"], (grams * f["mg_per_kg"] + 500) // 1000)

    def _slot(self, pid, vid, date, time, typ, ignore=""):
        p, v = self.state["patients"][pid], self._vet(vid)
        d = _pdate(date)
        if d < self._today():
            raise _Rule("%s is in the past (today is %s)" % (date, self.state["today"]))
        day = _DAY3[d.weekday()]
        hrs = self.state["policy"]["hours"].get(day)
        if not hrs:
            raise _Rule("the clinic is closed on %ss (%s)" % (_DAYNAMES[d.weekday()], date))
        start, dur = _ptime(time), self._dur(typ)
        if start % 5:
            raise _Rule("start times must be on a 5-minute mark, got %s" % time)
        if start < _ptime(hrs[0]) or start + dur > _ptime(hrs[1]):
            raise _Rule("a %d-minute %s starting %s does not fit the opening hours on %s (%s-%s)" % (
                dur, typ, time, date, hrs[0], hrs[1]))
        if day not in v["days"]:
            raise _Rule("%s (%s) does not work on %ss" % (vid, v["name"], _DAYNAMES[d.weekday()]))
        if p["species"] not in v["species"]:
            raise _Rule("%s (%s) does not treat %ss" % (vid, v["name"], p["species"]))
        if typ != "followup" and typ not in v["types"]:
            raise _Rule("%s (%s) does not do %s appointments" % (vid, v["name"], typ))
        for k, ap in sorted(self.state["appointments"].items()):
            if k == ignore or ap["status"] != "booked" or ap["date"] != date:
                continue
            if ap["vet"] != vid and ap["patient"] != pid:
                continue
            s2 = _ptime(ap["time"])
            e2 = s2 + self._dur(ap["type"])
            if start < e2 and s2 < start + dur:
                who = ("%s (%s)" % (vid, v["name"])) if ap["vet"] == vid else ("patient %s" % pid)
                raise _Rule("%s is already booked %s-%s on %s (%s)" % (who, ap["time"], _fmt_time(e2), date, k))
        if typ in ("dental", "surgery"):
            need = _plus_days(date, -self.state["policy"]["weight_valid_days"])
            wid, w = self._latest_weight(pid)
            if w is None or w["date"] < need:
                raise _Rule("a %s needs a weight recorded on or after %s (%d days before the appointment); the latest "
                            "weight for %s is from %s" % (typ, need, self.state["policy"]["weight_valid_days"], pid,
                                                          w["date"] if w else "never"))

    def _appt_row(self, k):
        ap = self.state["appointments"][k]
        p = self.state["patients"][ap["patient"]]
        return {"appointment_id": k, "patient_id": ap["patient"], "patient_name": p["name"], "species": p["species"],
                "vet_id": ap["vet"], "date": ap["date"], "time": ap["time"], "type": ap["type"],
                "minutes": self._dur(ap["type"]), "status": ap["status"], "reason": ap["reason"],
                "reminders": ap["reminders"], "followup_of": ap["followup_of"], "cancel_reason": ap["cancel_reason"]}

    def _t_get_clinic_policy(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "weekday": _DAYNAMES[self._today().weekday()],
                "opening_hours": p["hours"], "closed": "Sundays",
                "appointment_minutes": p["durations"],
                "vets": [dict(vet_id=k, **v) for k, v in sorted(self.state["vets"].items())],
                "vaccine_schedule": p["vaccines"], "formulary": p["formulary"],
                "rules": [
                    "Appointments start on a 5-minute mark, must fit inside the opening hours of that day, may not "
                    "be in the past, and need a vet who works that weekday, treats the species and does that type "
                    "of appointment. Neither the vet nor the patient may have an overlapping booked appointment.",
                    "Dental and surgery appointments need a weight recorded no more than %d days before the "
                    "appointment date. This is checked again when such an appointment is moved." % p["weight_valid_days"],
                    "Follow-ups (15 minutes, with the vet of the original visit) can only be booked for completed "
                    "appointments, for a date after the original visit and at most %d days after it; one follow-up "
                    "per visit." % p["followup_window_days"],
                    "A vaccine's next due date is the date given plus the schedule's booster_days. A vaccine may only "
                    "be given to the species listed, by a vet who treats that species, from the minimum age in "
                    "weeks, by one of its allowed routes, and not within 21 days of the same vaccine.",
                    "Prescribing needs a weight recorded within the last %d days (counting back from today). The "
                    "single dose may not exceed the formulary dose for that weight (mg_per_kg times kg, rounded to "
                    "the nearest mg, capped at max_mg), times_per_day may not exceed max_times_per_day, the species "
                    "must be listed for the drug and the patient may not be allergic to it. A patient may have only "
                    "one active prescription per drug." % p["weight_valid_days"],
                    "A drug cannot be added to a patient's allergy list while the patient has an active prescription "
                    "for it.",
                    "A patient can only be marked deceased or transferred when none of its appointments is still "
                    "booked and none of its prescriptions is active. Records that are not active cannot be changed.",
                ]}

    def _t_find_patients(self, a):
        q = a["query"].strip().lower()
        if not q:
            raise _Rule("query must not be empty")
        out = []
        for k, p in sorted(self.state["patients"].items()):
            owner = self.state["owners"][p["owner"]]["name"]
            if (q in p["name"].lower() or q in owner.lower()) and a.get("species", p["species"]) == p["species"]:
                out.append({"patient_id": k, "name": p["name"], "species": p["species"], "breed": p["breed"],
                            "owner": owner, "status": p["status"]})
        return {"patients": out}

    def _t_get_patient(self, a):
        pid = a["patient_id"]
        p = self._get("patients", pid, "patient")
        row = dict(patient_id=pid, **p)
        row["owner_name"] = self.state["owners"][p["owner"]]["name"]
        row["weights"] = [dict(weight_id=k, date=w["date"], grams=w["grams"], body_condition=w["body_condition"])
                          for _, k, w in sorted((w["date"], k, w) for k, w in self.state["weights"].items()
                                                if w["patient"] == pid)]
        row["vaccinations"] = [dict(vaccination_id=k, **{f: v[f] for f in ("vaccine", "date_given", "next_due",
                                                                              "route", "vet")})
                               for k, v in sorted(self.state["vaccinations"].items()) if v["patient"] == pid]
        row["prescriptions"] = [dict(prescription_id=k, **r) for k, r in sorted(self.state["prescriptions"].items())
                                if r["patient"] == pid]
        row["booked_appointments"] = [self._appt_row(k) for k in sorted(
            self.state["appointments"], key=lambda k: (self.state["appointments"][k]["date"],
                                                       self.state["appointments"][k]["time"], k))
            if self.state["appointments"][k]["patient"] == pid and self.state["appointments"][k]["status"] == "booked"]
        return row

    def _t_list_appointments(self, a):
        if "vet_id" in a:
            self._vet(a["vet_id"])
        if "patient_id" in a:
            self._get("patients", a["patient_id"], "patient")
        if "date" in a:
            _pdate(a["date"])
        rows = [k for _, _, k in sorted((ap["date"], ap["time"], k) for k, ap in self.state["appointments"].items()
                                        if ("date" not in a or ap["date"] == a["date"])
                                        and ("vet_id" not in a or ap["vet"] == a["vet_id"])
                                        and ("patient_id" not in a or ap["patient"] == a["patient_id"])
                                        and ("status" not in a or ap["status"] == a["status"]))]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["appointments"] = [self._appt_row(k) for k in rows]
        return meta

    def _t_get_dosing(self, a):
        f = self.state["policy"]["formulary"][a["drug"]]
        if a["species"] not in f["species"]:
            return {"drug": a["drug"], "species": a["species"], "licensed_for_species": False}
        return {"drug": a["drug"], "species": a["species"], "licensed_for_species": True,
                "weight_kg": a["weight_grams"] / 1000.0, "mg_per_kg": f["mg_per_kg"],
                "dose_mg": self._dose(a["drug"], a["weight_grams"]),
                "capped_at_max": a["weight_grams"] * f["mg_per_kg"] / 1000.0 > f["max_mg"],
                "max_times_per_day": f["max_times_per_day"]}

    def _reminders(self, rs):
        seen = set()
        for r in rs:
            if (r["channel"], r["days_before"]) in seen:
                raise _Rule("the same reminder is listed twice")
            seen.add((r["channel"], r["days_before"]))
        return [{"channel": c, "days_before": d} for d, c in sorted((r["days_before"], r["channel"]) for r in rs)]

    def _t_schedule_appointment(self, a):
        pid, vid = a["patient_id"], a["vet_id"]
        self._patient(pid)
        self._vet(vid)
        self._slot(pid, vid, a["date"], a["time"], a["type"])
        k = self._new_id("next_appt", "AP")
        self.state["appointments"][k] = {
            "patient": pid, "vet": vid, "date": a["date"], "time": a["time"], "type": a["type"], "status": "booked",
            "reason": _text(a["reason"], "reason") if "reason" in a else "",
            "reminders": self._reminders(a.get("reminders", [])), "followup_of": "", "cancel_reason": ""}
        return {"ok": True, "appointment": self._appt_row(k)}

    def _t_reschedule_appointment(self, a):
        k = a["appointment_id"]
        ap = self._get("appointments", k, "appointment")
        if ap["status"] != "booked":
            raise _Rule("appointment %s is %s; only booked appointments can be moved" % (k, ap["status"]))
        self._patient(ap["patient"])
        vid = a.get("vet_id", ap["vet"])
        self._vet(vid)
        if ap["followup_of"]:
            orig = self.state["appointments"][ap["followup_of"]]
            if vid != orig["vet"]:
                raise _Rule("a follow-up stays with the vet of the original visit (%s)" % orig["vet"])
            self._followup_window(orig, a["date"])
        if (a["date"], a["time"], vid) == (ap["date"], ap["time"], ap["vet"]):
            raise _Rule("appointment %s is already at %s %s with %s" % (k, ap["date"], ap["time"], vid))
        self._slot(ap["patient"], vid, a["date"], a["time"], ap["type"], ignore=k)
        ap["date"], ap["time"], ap["vet"] = a["date"], a["time"], vid
        return {"ok": True, "appointment": self._appt_row(k)}

    def _followup_window(self, orig, date):
        last = _plus_days(orig["date"], self.state["policy"]["followup_window_days"])
        if not (orig["date"] < date <= last):
            raise _Rule("a follow-up of the visit on %s must be after it and no later than %s" % (orig["date"], last))

    def _t_book_followup(self, a):
        k = a["appointment_id"]
        orig = self._get("appointments", k, "appointment")
        if orig["status"] != "completed":
            raise _Rule("appointment %s is %s; follow-ups can only be booked for completed visits" % (k, orig["status"]))
        self._patient(orig["patient"])
        _pdate(a["date"])
        self._followup_window(orig, a["date"])
        for k2, ap in sorted(self.state["appointments"].items()):
            if ap["followup_of"] == k and ap["status"] == "booked":
                raise _Rule("appointment %s already has a booked follow-up (%s)" % (k, k2))
        self._slot(orig["patient"], orig["vet"], a["date"], a["time"], "followup")
        n = self._new_id("next_appt", "AP")
        self.state["appointments"][n] = {
            "patient": orig["patient"], "vet": orig["vet"], "date": a["date"], "time": a["time"], "type": "followup",
            "status": "booked", "reason": _text(a["reason"], "reason") if "reason" in a else "", "reminders": [],
            "followup_of": k, "cancel_reason": ""}
        return {"ok": True, "appointment": self._appt_row(n)}

    def _t_cancel_appointment(self, a):
        k = a["appointment_id"]
        ap = self._get("appointments", k, "appointment")
        if ap["status"] != "booked":
            raise _Rule("appointment %s is %s, not booked" % (k, ap["status"]))
        ap["status"], ap["cancel_reason"] = "cancelled", a["reason"]
        return {"ok": True, "appointment": self._appt_row(k)}

    def _t_record_vaccinations(self, a):
        pid, vid = a["patient_id"], a["vet_id"]
        p = self._patient(pid)
        v = self._vet(vid)
        d = _pdate(a["date_given"], "date_given")
        if d > self._today():
            raise _Rule("date_given %s is in the future (today is %s)" % (a["date_given"], self.state["today"]))
        if p["species"] not in v["species"]:
            raise _Rule("%s (%s) does not treat %ss" % (vid, v["name"], p["species"]))
        age = (d - _pdate(p["dob"])).days
        seen = set()
        for item in a["vaccines"]:
            vac = item["vaccine"]
            sch = self.state["policy"]["vaccines"][vac]
            if vac in seen:
                raise _Rule("%s is listed twice" % vac)
            seen.add(vac)
            if p["species"] not in sch["species"]:
                raise _Rule("%s is not given to %ss" % (vac, p["species"]))
            if age < sch["min_age_weeks"] * 7:
                raise _Rule("%s is %d days old on %s; %s needs at least %d weeks" % (
                    p["name"], age, a["date_given"], vac, sch["min_age_weeks"]))
            route = item.get("route", "subcutaneous")
            if route not in sch["routes"]:
                raise _Rule("%s cannot be given %s (allowed: %s)" % (vac, route, ", ".join(sch["routes"])))
            for k, old in sorted(self.state["vaccinations"].items()):
                if old["patient"] == pid and old["vaccine"] == vac and \
                        abs(_days_between(old["date_given"], a["date_given"])) < 21:
                    raise _Rule("%s already had %s on %s (%s)" % (pid, vac, old["date_given"], k))
        made = []
        for item in a["vaccines"]:
            sch = self.state["policy"]["vaccines"][item["vaccine"]]
            k = self._new_id("next_vax", "VX")
            self.state["vaccinations"][k] = {
                "patient": pid, "vaccine": item["vaccine"], "date_given": a["date_given"], "vet": vid,
                "batch": _text(item["batch_number"], "batch_number", 20), "route": item.get("route", "subcutaneous"),
                "next_due": _plus_days(a["date_given"], sch["booster_days"])}
            made.append(dict(vaccination_id=k, **self.state["vaccinations"][k]))
        return {"ok": True, "vaccinations": made}

    def _t_record_weight(self, a):
        pid = a["patient_id"]
        p = self._patient(pid)
        d = _pdate(a["date"])
        if d > self._today():
            raise _Rule("%s is in the future (today is %s)" % (a["date"], self.state["today"]))
        if a["date"] < p["dob"]:
            raise _Rule("%s is before %s was born" % (a["date"], p["name"]))
        k = self._new_id("next_weight", "WT")
        self.state["weights"][k] = {"patient": pid, "date": a["date"], "grams": a["weight_grams"],
                                    "body_condition": a.get("body_condition", 0)}
        return {"ok": True, "weight_id": k, "weight": self.state["weights"][k]}

    def _t_prescribe_medication(self, a):
        pid, vid, drug = a["patient_id"], a["vet_id"], a["drug"]
        p = self._patient(pid)
        v = self._vet(vid)
        f = self.state["policy"]["formulary"][drug]
        if p["species"] not in v["species"]:
            raise _Rule("%s (%s) does not treat %ss" % (vid, v["name"], p["species"]))
        if p["species"] not in f["species"]:
            raise _Rule("%s is not licensed for %ss" % (drug, p["species"]))
        if drug in p["allergies"]:
            raise _Rule("%s is recorded as allergic to %s" % (p["name"], drug))
        wid, w = self._latest_weight(pid)
        need = _plus_days(self.state["today"], -self.state["policy"]["weight_valid_days"])
        if w is None or w["date"] < need:
            raise _Rule("prescribing needs a weight recorded on or after %s; the latest weight for %s is from %s" % (
                need, pid, w["date"] if w else "never"))
        dose = self._dose(drug, w["grams"])
        if a["dose_mg"] > dose:
            raise _Rule("%d mg is above the formulary dose of %d mg for %s at %d g" % (a["dose_mg"], dose, drug, w["grams"]))
        if a["times_per_day"] > f["max_times_per_day"]:
            raise _Rule("%s may be given at most %d times a day" % (drug, f["max_times_per_day"]))
        for k, r in sorted(self.state["prescriptions"].items()):
            if r["patient"] == pid and r["drug"] == drug and r["status"] == "active":
                raise _Rule("%s already has an active %s prescription (%s); stop it first" % (pid, drug, k))
        k = self._new_id("next_rx", "RX")
        self.state["prescriptions"][k] = {
            "patient": pid, "drug": drug, "dose_mg": a["dose_mg"], "times_per_day": a["times_per_day"],
            "days": a["days"], "vet": vid, "start": self.state["today"], "with_food": a.get("with_food", False),
            "status": "active", "stopped": "", "stop_reason": ""}
        return {"ok": True, "prescription": dict(prescription_id=k, **self.state["prescriptions"][k])}

    def _t_stop_prescription(self, a):
        k = a["prescription_id"]
        r = self._get("prescriptions", k, "prescription")
        if r["status"] != "active":
            raise _Rule("prescription %s is already %s" % (k, r["status"]))
        r["status"], r["stopped"], r["stop_reason"] = "stopped", self.state["today"], a["reason"]
        return {"ok": True, "prescription": dict(prescription_id=k, **r)}

    def _t_update_patient(self, a):
        pid = a["patient_id"]
        p = self._patient(pid)
        if len(a) < 2:
            raise _Rule("nothing to change: give microchip, neutered, allergies or status")
        if "microchip" in a:
            if not re.match(r"^\d{15}$", a["microchip"]):
                raise _Rule("a microchip number has exactly 15 digits")
            p["microchip"] = a["microchip"]
        if "neutered" in a:
            if p["neutered"] and not a["neutered"]:
                raise _Rule("neutering cannot be undone")
            p["neutered"] = a["neutered"]
        if "allergies" in a:
            new = sorted(set(a["allergies"]))
            for k, r in sorted(self.state["prescriptions"].items()):
                if r["patient"] == pid and r["status"] == "active" and r["drug"] in new:
                    raise _Rule("%s has an active %s prescription (%s); stop it first" % (pid, r["drug"], k))
            p["allergies"] = new
        if "status" in a and a["status"] != "active":
            for k, ap in sorted(self.state["appointments"].items()):
                if ap["patient"] == pid and ap["status"] == "booked":
                    raise _Rule("%s still has a booked appointment (%s on %s)" % (pid, k, ap["date"]))
            for k, r in sorted(self.state["prescriptions"].items()):
                if r["patient"] == pid and r["status"] == "active":
                    raise _Rule("%s still has an active prescription (%s, %s)" % (pid, k, r["drug"]))
            p["status"] = a["status"]
        return {"ok": True, "patient": self._t_get_patient({"patient_id": pid})}


def _vet():
    days5 = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    vets = {
        "V-01": {"name": "Dr Amaya Ruiz", "species": ["dog", "cat"], "types": ["consult", "vaccination", "dental", "surgery"],
                 "days": days5},
        "V-02": {"name": "Dr Kofi Mensah", "species": ["dog", "cat"], "types": ["consult", "vaccination", "nail_trim"],
                 "days": ["Mon", "Wed", "Fri", "Sat"]},
        "V-03": {"name": "Dr Lena Hart", "species": ["rabbit", "ferret", "bird", "cat"],
                 "types": ["consult", "vaccination", "dental", "surgery", "nail_trim"], "days": ["Tue", "Thu", "Sat"]},
        "V-04": {"name": "Dr Piotr Nowak", "species": ["dog", "cat", "rabbit"], "types": ["consult", "dental", "surgery"],
                 "days": ["Mon", "Tue", "Wed", "Thu"]},
        "V-05": {"name": "Dr Hana Sato", "species": ["dog", "cat", "ferret"], "types": ["consult", "vaccination", "nail_trim"],
                 "days": ["Tue", "Wed", "Thu", "Fri", "Sat"]},
    }
    owners = dict((k, {"name": n, "phone": ph}) for k, n, ph in [
        ("O-01", "Daniel Haddad", "0161 555 0142"), ("O-02", "Ines Okafor", "0161 555 0177"),
        ("O-03", "Marta Silva", "0161 555 0198"), ("O-04", "Tom Becker", "0161 555 0123"),
        ("O-05", "Grace Liu", "0161 555 0110"), ("O-06", "Ana Petrova", "0161 555 0164"),
        ("O-07", "Sam Wilder", "0161 555 0181"), ("O-08", "Yara Haddad", "0161 555 0136"),
        ("O-09", "Paul Okafor", "0161 555 0159"), ("O-10", "Erik Lund", "0161 555 0105")])
    patients = {}
    for pid, name, sp, breed, sex, dob, owner, neut, allergies, status in [
        ("A-101", "Rex", "dog", "Labrador", "male", "2019-03-14", "O-01", True, [], "active"),
        ("A-102", "Mochi", "cat", "British Shorthair", "female", "2021-06-02", "O-02", True, [], "active"),
        ("A-103", "Biscuit", "dog", "Beagle", "female", "2022-01-20", "O-03", True, [], "active"),
        ("A-104", "Juniper", "rabbit", "Netherland Dwarf", "female", "2023-04-11", "O-04", True, [], "active"),
        ("A-105", "Bruno", "dog", "Boxer", "male", "2020-08-30", "O-05", False, [], "active"),
        ("A-106", "Pepper", "dog", "Cocker Spaniel", "female", "2018-11-05", "O-01", True, [], "active"),
        ("A-107", "Luna", "cat", "Siamese", "female", "2017-02-14", "O-06", True, ["cefalexin"], "active"),
        ("A-108", "Luna", "dog", "Husky", "female", "2021-09-09", "O-07", False, [], "active"),
        ("A-109", "Ziggy", "ferret", "Sable", "male", "2023-07-01", "O-08", True, [], "active"),
        ("A-110", "Clover", "rabbit", "Lop", "female", "2025-12-20", "O-04", False, [], "active"),
        ("A-111", "Nori", "cat", "Domestic Shorthair", "male", "2026-09-10", "O-02", False, [], "active"),
        ("A-112", "Pepper", "cat", "Maine Coon", "male", "2016-05-05", "O-09", True, ["amoxicillin"], "active"),
        ("A-113", "Odin", "dog", "Great Dane", "male", "2020-02-02", "O-10", True, [], "active"),
        ("A-114", "Saffron", "bird", "Cockatiel", "female", "2019-03-03", "O-06", False, [], "active"),
        ("A-115", "Tilly", "dog", "Jack Russell", "female", "2026-08-15", "O-02", False, [], "active"),
        ("A-116", "Basil", "cat", "Tabby", "male", "2010-10-10", "O-03", True, [], "deceased"),
        ("A-117", "Maple", "dog", "Border Collie", "female", "2016-01-01", "O-05", True, [], "active"),
        ("A-118", "Kiwi", "rabbit", "Rex", "male", "2022-05-05", "O-08", True, [], "active"),
        ("A-119", "Mochi", "dog", "Shiba Inu", "male", "2024-02-02", "O-09", False, [], "active"),
    ]:
        patients[pid] = {"name": name, "species": sp, "breed": breed, "sex": sex, "dob": dob, "owner": owner,
                         "neutered": neut, "microchip": "", "allergies": allergies, "status": status}
    patients["A-101"]["microchip"] = "985112004417730"
    weights = {}
    for k, pid, date, g, bcs in [
        ("WT-701", "A-101", "2026-11-04", 32400, 5), ("WT-702", "A-101", "2026-06-01", 33100, 6),
        ("WT-703", "A-102", "2026-10-26", 4300, 5), ("WT-704", "A-103", "2026-08-12", 10950, 6),
        ("WT-705", "A-105", "2026-10-20", 29800, 5), ("WT-706", "A-106", "2026-09-01", 20000, 7),
        ("WT-707", "A-107", "2026-10-30", 3900, 4), ("WT-708", "A-109", "2026-10-15", 1120, 5),
        ("WT-709", "A-113", "2026-09-01", 63500, 6), ("WT-710", "A-104", "2026-10-01", 1900, 5),
        ("WT-711", "A-115", "2026-11-02", 4100, 5), ("WT-712", "A-111", "2026-11-02", 1000, 5),
        ("WT-713", "A-117", "2026-10-28", 24100, 4), ("WT-714", "A-114", "2026-07-07", 95, 5),
        ("WT-715", "A-108", "2026-10-25", 23000, 5), ("WT-716", "A-112", "2026-10-12", 7200, 6),
    ]:
        weights[k] = {"patient": pid, "date": date, "grams": g, "body_condition": bcs}
    vac_policy = {
        "dhpp": {"species": ["dog"], "booster_days": 1095, "min_age_weeks": 6, "routes": ["subcutaneous"],
                 "covers": "distemper, hepatitis, parvovirus and parainfluenza"},
        "lepto": {"species": ["dog"], "booster_days": 365, "min_age_weeks": 8,
                  "routes": ["subcutaneous", "intramuscular"], "covers": "leptospirosis"},
        "kennel_cough": {"species": ["dog"], "booster_days": 365, "min_age_weeks": 3,
                         "routes": ["intranasal", "oral", "subcutaneous"], "covers": "Bordetella (kennel cough)"},
        "rabies": {"species": ["dog", "cat", "ferret"], "booster_days": 365, "min_age_weeks": 12,
                   "routes": ["subcutaneous", "intramuscular"], "covers": "rabies"},
        "fvrcp": {"species": ["cat"], "booster_days": 1095, "min_age_weeks": 6, "routes": ["subcutaneous"],
                  "covers": "feline rhinotracheitis, calicivirus and panleukopenia"},
        "felv": {"species": ["cat"], "booster_days": 365, "min_age_weeks": 8, "routes": ["subcutaneous"],
                 "covers": "feline leukaemia virus"},
        "rhd": {"species": ["rabbit"], "booster_days": 365, "min_age_weeks": 5, "routes": ["subcutaneous"],
                "covers": "rabbit haemorrhagic disease"},
        "myxo": {"species": ["rabbit"], "booster_days": 365, "min_age_weeks": 5, "routes": ["subcutaneous"],
                 "covers": "myxomatosis"},
    }
    vaccinations = {}
    for k, pid, vac, date, vet in [
        ("VX-501", "A-101", "dhpp", "2024-03-10", "V-01"), ("VX-502", "A-101", "rabies", "2025-11-20", "V-01"),
        ("VX-503", "A-102", "fvrcp", "2024-05-15", "V-02"), ("VX-504", "A-102", "felv", "2025-10-30", "V-02"),
        ("VX-505", "A-102", "rabies", "2025-12-01", "V-05"), ("VX-506", "A-105", "dhpp", "2023-11-01", "V-02"),
        ("VX-507", "A-105", "lepto", "2025-11-01", "V-02"), ("VX-508", "A-115", "dhpp", "2026-10-10", "V-05"),
        ("VX-509", "A-104", "rhd", "2025-11-15", "V-03"), ("VX-510", "A-104", "myxo", "2025-11-15", "V-03"),
        ("VX-511", "A-107", "fvrcp", "2025-02-01", "V-01"), ("VX-512", "A-117", "rabies", "2026-03-03", "V-04"),
        ("VX-513", "A-109", "rabies", "2026-01-20", "V-03"), ("VX-514", "A-112", "felv", "2026-02-02", "V-05"),
        ("VX-515", "A-119", "dhpp", "2024-04-04", "V-02"), ("VX-516", "A-112", "rabies", "2025-11-30", "V-05"),
    ]:
        vaccinations[k] = {"patient": pid, "vaccine": vac, "date_given": date, "vet": vet,
                           "batch": "B%s" % k[3:], "route": "subcutaneous",
                           "next_due": _plus_days(date, vac_policy[vac]["booster_days"])}
    prescriptions = {}
    for k, pid, drug, dose, tpd, days, vet, start, food, status, stopped, why in [
        ("RX-901", "A-106", "amoxicillin", 400, 2, 10, "V-01", "2026-11-05", True, "active", "", ""),
        ("RX-902", "A-117", "gabapentin", 240, 2, 30, "V-04", "2026-10-28", False, "active", "", ""),
        ("RX-903", "A-117", "carprofen", 90, 1, 30, "V-04", "2026-10-01", True, "stopped", "2026-10-28", "course_complete"),
        ("RX-904", "A-107", "gabapentin", 40, 2, 14, "V-02", "2026-11-03", False, "active", "", ""),
        ("RX-905", "A-101", "maropitant", 60, 1, 3, "V-01", "2026-09-01", False, "stopped", "2026-09-04", "course_complete"),
        ("RX-906", "A-108", "cefalexin", 345, 2, 7, "V-02", "2026-11-06", True, "active", "", ""),
    ]:
        prescriptions[k] = {"patient": pid, "drug": drug, "dose_mg": dose, "times_per_day": tpd, "days": days,
                            "vet": vet, "start": start, "with_food": food, "status": status, "stopped": stopped,
                            "stop_reason": why}
    appointments = {}
    for k, pid, vet, date, time, typ, status, fup in [
        ("AP-3001", "A-101", "V-01", "2026-11-02", "09:00", "consult", "completed", ""),
        ("AP-3004", "A-102", "V-01", "2026-11-02", "10:00", "dental", "completed", ""),
        ("AP-3005", "A-117", "V-04", "2026-11-12", "14:00", "consult", "booked", ""),
        ("AP-3006", "A-117", "V-01", "2026-11-20", "09:30", "consult", "booked", ""),
        ("AP-3007", "A-117", "V-04", "2026-10-28", "11:00", "consult", "completed", ""),
        ("AP-3009", "A-108", "V-02", "2026-11-11", "15:00", "consult", "booked", ""),
        ("AP-3010", "A-104", "V-03", "2026-11-10", "09:30", "consult", "booked", ""),
        ("AP-3011", "A-109", "V-03", "2026-11-12", "09:00", "surgery", "booked", ""),
        ("AP-3012", "A-104", "V-03", "2026-11-21", "10:30", "nail_trim", "booked", ""),
        ("AP-3013", "A-118", "V-03", "2026-11-21", "12:40", "consult", "booked", ""),
        ("AP-3014", "A-113", "V-04", "2026-10-20", "15:00", "surgery", "completed", ""),
        ("AP-3015", "A-113", "V-04", "2026-10-30", "15:00", "followup", "completed", "AP-3014"),
        ("AP-3020", "A-105", "V-01", "2026-11-18", "09:50", "consult", "booked", ""),
        ("AP-3021", "A-103", "V-04", "2026-11-18", "08:30", "consult", "booked", ""),
        ("AP-3030", "A-105", "V-02", "2026-11-13", "09:00", "consult", "booked", ""),
        ("AP-3031", "A-102", "V-02", "2026-11-13", "09:30", "vaccination", "booked", ""),
        ("AP-3032", "A-103", "V-02", "2026-11-13", "10:00", "nail_trim", "booked", ""),
        ("AP-3033", "A-119", "V-02", "2026-11-13", "11:00", "consult", "booked", ""),
        ("AP-3034", "A-108", "V-05", "2026-11-13", "09:55", "consult", "booked", ""),
        ("AP-3035", "A-101", "V-02", "2026-11-13", "12:00", "consult", "cancelled", ""),
        ("AP-3036", "A-107", "V-05", "2026-11-13", "11:20", "consult", "booked", ""),
        ("AP-3040", "A-110", "V-03", "2026-11-14", "09:00", "vaccination", "booked", ""),
        ("AP-3041", "A-106", "V-02", "2026-11-09", "16:00", "consult", "booked", ""),
        ("AP-3042", "A-112", "V-05", "2026-11-10", "10:00", "consult", "booked", ""),
        ("AP-3045", "A-105", "V-02", "2026-11-14", "11:00", "nail_trim", "booked", ""),
        ("AP-3046", "A-112", "V-05", "2026-11-14", "10:00", "consult", "booked", ""),
        ("AP-3047", "A-102", "V-02", "2026-11-14", "11:30", "consult", "cancelled", ""),
        ("AP-3048", "A-103", "V-02", "2026-11-16", "10:50", "consult", "booked", ""),
        ("AP-3050", "A-112", "V-01", "2026-11-17", "09:50", "consult", "booked", ""),
        ("AP-3051", "A-118", "V-03", "2026-11-14", "12:00", "nail_trim", "booked", ""),
    ]:
        appointments[k] = {"patient": pid, "vet": vet, "date": date, "time": time, "type": typ, "status": status,
                           "reason": "", "reminders": [], "followup_of": fup,
                           "cancel_reason": "owner_request" if status == "cancelled" else ""}
    st = {"today": "2026-11-09",
          "policy": {"hours": {"Mon": ["08:00", "18:00"], "Tue": ["08:00", "18:00"], "Wed": ["08:00", "18:00"],
                               "Thu": ["08:00", "18:00"], "Fri": ["08:00", "18:00"], "Sat": ["09:00", "13:00"]},
                     "durations": {"consult": 20, "vaccination": 15, "dental": 60, "surgery": 120, "nail_trim": 10,
                                   "followup": 15},
                     "weight_valid_days": 30, "followup_window_days": 28, "vaccines": vac_policy,
                     "formulary": {
                         "amoxicillin": {"species": ["dog", "cat"], "mg_per_kg": 20, "max_mg": 1000, "max_times_per_day": 2},
                         "cefalexin": {"species": ["dog", "cat"], "mg_per_kg": 15, "max_mg": 1500, "max_times_per_day": 2},
                         "carprofen": {"species": ["dog"], "mg_per_kg": 4, "max_mg": 300, "max_times_per_day": 2},
                         "gabapentin": {"species": ["dog", "cat"], "mg_per_kg": 10, "max_mg": 600, "max_times_per_day": 3},
                         "maropitant": {"species": ["dog", "cat"], "mg_per_kg": 2, "max_mg": 160, "max_times_per_day": 1},
                         "enrofloxacin": {"species": ["dog", "cat", "rabbit"], "mg_per_kg": 5, "max_mg": 500,
                                          "max_times_per_day": 1},
                         "prednisolone": {"species": ["dog", "cat"], "mg_per_kg": 1, "max_mg": 40, "max_times_per_day": 2}}},
          "vets": vets, "owners": owners, "patients": patients, "appointments": appointments,
          "vaccinations": vaccinations, "weights": weights, "prescriptions": prescriptions,
          "next_appt": 3100, "next_vax": 600, "next_weight": 800, "next_rx": 950}
    for k, ap in appointments.items():
        assert ap["patient"] in patients and ap["vet"] in vets, k
        assert patients[ap["patient"]]["species"] in vets[ap["vet"]]["species"], k
    return st


def _appt(pid, vet, date, time, typ, reminders=(), followup_of=""):
    return {"patient": pid, "vet": vet, "date": date, "time": time, "type": typ, "status": "booked",
            "reason": _anystr, "reminders": list(reminders), "followup_of": followup_of, "cancel_reason": ""}


def _wt(pid, date, grams, bcs=0):
    return {"patient": pid, "date": date, "grams": grams, "body_condition": bcs}


def _vx(pid, vac, date, vet, batch, route="subcutaneous", booster=365):
    return {"patient": pid, "vaccine": vac, "date_given": date, "vet": vet, "batch": batch, "route": route,
            "next_due": _plus_days(date, booster)}


def _rx(pid, drug, dose, tpd, days, vet, food, start="2026-11-09"):
    return {"patient": pid, "drug": drug, "dose_mg": dose, "times_per_day": tpd, "days": days, "vet": vet,
            "start": start, "with_food": food, "status": "active", "stopped": "", "stop_reason": ""}


_VT = "2026-11-09"   # vet "today"

# ---- single-call S01-S06
_add("S01", "vet", _vet(),
     "Front-desk note for Biscuit the beagle (patient A-103, owner Marta Silva, phone 0161 555 0198, account opened "
     "2022). She was weighed this morning, 9 November 2026, on the old reception scale, which reads in pounds and "
     "ounces: 23 lb 6 oz. Convert it to grams using 1 lb = 453.59 g and 1 oz = 28.35 g, round to the nearest gram, "
     "and log the weighing. The nurse also scored her body condition at 6 out of 9.",
     [["record_weight", {"patient_id": "A-103", "date": "2026-11-09", "weight_grams": 10603, "body_condition": 6}]],
     [["record_weight", {"patient_id": "A-103", "date": "2026-11-09", "weight_grams": 10603}]],
     lambda s, i: _final(s, i, new={"weights": [_wt("A-103", "2026-11-09", 10603, 6)]}),
     "UPDX", mentions=["Biscuit"])

_add("S02", "vet", _vet(),
     "Mochi the British Shorthair (A-102) had her dental with Dr Ruiz on Monday 2 November 2026 - that visit is "
     "appointment AP-3004 and it is marked completed. Book her post-dental check as a follow-up of that visit, "
     "exactly two weeks after the dental, at 11:20.",
     [["book_followup", {"appointment_id": "AP-3004", "date": "2026-11-16", "time": "11:20"}]],
     [["schedule_appointment", {"patient_id": "A-102", "vet_id": "V-01", "date": "2026-11-16", "time": "11:20",
                                "type": "consult"}]],
     lambda s, i: _final(s, i, new={"appointments": [_appt("A-102", "V-01", "2026-11-16", "11:20", "followup",
                                                           followup_of="AP-3004")]}),
     "TD", mentions=["Mochi", "Ruiz"])

_add("S03", "vet", _vet(),
     "Bruno the boxer (A-105) came in today, 9 November 2026, and Dr Mensah (V-02) gave him three vaccines: the "
     "combined distemper / hepatitis / parvovirus / parainfluenza booster (batch DH-7781), his leptospirosis "
     "booster (batch LP-2093), and the kennel cough vaccine, squirted up his nose (batch KC-4410). Record all "
     "three. The first two went in under the skin as usual.",
     [["record_vaccinations", {"patient_id": "A-105", "date_given": "2026-11-09", "vet_id": "V-02", "vaccines": [
         {"vaccine": "dhpp", "batch_number": "DH-7781"}, {"vaccine": "lepto", "batch_number": "LP-2093"},
         {"vaccine": "kennel_cough", "batch_number": "KC-4410", "route": "intranasal"}]}]],
     [["record_vaccinations", {"patient_id": "A-105", "date_given": "2026-11-09", "vet_id": "V-02", "vaccines": [
         {"vaccine": "dhpp", "batch_number": "DH-7781"}, {"vaccine": "lepto", "batch_number": "LP-2093"},
         {"vaccine": "kennel_cough", "batch_number": "KC-4410"}]}]],
     lambda s, i: _final(s, i, new={"vaccinations": [
         _vx("A-105", "dhpp", _VT, "V-02", "DH-7781", booster=1095), _vx("A-105", "lepto", _VT, "V-02", "LP-2093"),
         _vx("A-105", "kennel_cough", _VT, "V-02", "KC-4410", route="intranasal")]}),
     "ONPD", mentions=["Bruno", "Mensah"], creates=["DH-7781", "LP-2093", "KC-4410"])

_add("S04", "vet", _vet(),
     "Juniper the rabbit (A-104) is booked with Dr Hart for a consult on Tuesday 10 November 2026 at 09:30 "
     "(appointment AP-3010). Her owner, Tom Becker, can't make Tuesday. Please move that appointment to the "
     "Thursday of the same week at half past eleven in the morning, still with Dr Hart.",
     [["reschedule_appointment", {"appointment_id": "AP-3010", "date": "2026-11-12", "time": "11:30"}]],
     [["schedule_appointment", {"patient_id": "A-104", "vet_id": "V-03", "date": "2026-11-12", "time": "11:30",
                                "type": "consult"}]],
     lambda s, i: _final(s, i, changes={"appointments": {"AP-3010": {"date": "2026-11-12", "time": "11:30"}}}),
     "TDU", mentions=["Juniper", "Hart", "Tom Becker"])

_add("S05", "vet", _vet(),
     "Dr Nowak (V-04) is starting Rex (A-101) on carprofen after this morning's exam. From the notes: Rex is "
     "32.4 kg (weighed on 4 November), temperature 38.6, heart rate 92, chip 985112004417730. The plan is 125 mg "
     "twice a day for 10 days, always given with food. His 2024 trial used 75 mg once a day - ignore that. "
     "Write the prescription.",
     [["prescribe_medication", {"patient_id": "A-101", "drug": "carprofen", "dose_mg": 125, "times_per_day": 2,
                                "days": 10, "vet_id": "V-04", "with_food": True}]],
     [["prescribe_medication", {"patient_id": "A-101", "drug": "carprofen", "dose_mg": 125, "times_per_day": 2,
                                "days": 10, "vet_id": "V-04"}]],
     lambda s, i: _final(s, i, new={"prescriptions": [_rx("A-101", "carprofen", 125, 2, 10, "V-04", True)]}),
     "XPN", mentions=["Rex", "Nowak"])

_add("S06", "vet", _vet(),
     "Luna the Siamese cat (A-107) has been vomiting after every dose of the gabapentin we started on 3 November "
     "(prescription RX-904), so her owner stopped giving it. Take her off that prescription and record why it was "
     "stopped.",
     [["stop_prescription", {"prescription_id": "RX-904", "reason": "adverse_reaction"}]],
     [["stop_prescription", {"prescription_id": "RX-904", "reason": "owner_request"}]],
     lambda s, i: _final(s, i, changes={"prescriptions": {"RX-904": {
         "status": "stopped", "stopped": _VT, "stop_reason": "adverse_reaction"}}}),
     "NT", mentions=["Luna"])

# ---- multi-step M01-M06
_add("M01", "vet", _vet(),
     "Odin the Great Dane needs a dental cleaning on Wednesday 18 November at 10:00. Book it with the "
     "lowest-numbered vet who does dentals on dogs, works that day and is free at that time. Our nurse weighed "
     "him this morning at 61.2 kg with a body condition of 6, but nobody has entered that on his record yet. "
     "Nothing else should change.",
     [["get_clinic_policy", {}], ["find_patients", {"query": "Odin"}],
      ["list_appointments", {"date": "2026-11-18"}],
      ["schedule_appointment", {"patient_id": "A-113", "vet_id": "V-04", "date": "2026-11-18", "time": "10:00",
                                "type": "dental"}],
      ["record_weight", {"patient_id": "A-113", "date": "2026-11-09", "weight_grams": 61200, "body_condition": 6}],
      ["get_patient", {"patient_id": "A-113"}],
      ["schedule_appointment", {"patient_id": "A-113", "vet_id": "V-04", "date": "2026-11-18", "time": "10:00",
                                "type": "dental"}]],
     [["schedule_appointment", {"patient_id": "A-113", "vet_id": "V-01", "date": "2026-11-18", "time": "10:00",
                                "type": "dental"}],
      ["record_weight", {"patient_id": "A-113", "date": "2026-11-09", "weight_grams": 61200, "body_condition": 6}],
      ["schedule_appointment", {"patient_id": "A-113", "vet_id": "V-01", "date": "2026-11-18", "time": "10:30",
                                "type": "dental"}]],
     lambda s, i: _final(s, i, new={"weights": [_wt("A-113", _VT, 61200, 6)],
                                    "appointments": [_appt("A-113", "V-04", "2026-11-18", "10:00", "dental")]}),
     "abcf", ref_errors=[3], mentions=["Odin"])

_add("M02", "vet", _vet(),
     "Sad news from Grace Liu: her old collie Maple died over the weekend. Please close Maple's record - cancel "
     "anything still booked for her, stop any medication she is on, and mark her as deceased, using the reasons "
     "that fit. Grace's other dog keeps all his appointments, and Maple's past visits and finished prescriptions "
     "stay as they are.",
     [["find_patients", {"query": "Maple"}], ["get_patient", {"patient_id": "A-117"}],
      ["update_patient", {"patient_id": "A-117", "status": "deceased"}],
      ["cancel_appointment", {"appointment_id": "AP-3005", "reason": "patient_deceased"}],
      ["cancel_appointment", {"appointment_id": "AP-3006", "reason": "patient_deceased"}],
      ["stop_prescription", {"prescription_id": "RX-902", "reason": "patient_deceased"}],
      ["get_patient", {"patient_id": "A-117"}],
      ["update_patient", {"patient_id": "A-117", "status": "deceased"}]],
     [["cancel_appointment", {"appointment_id": "AP-3005", "reason": "owner_request"}],
      ["cancel_appointment", {"appointment_id": "AP-3006", "reason": "owner_request"}],
      ["stop_prescription", {"prescription_id": "RX-902", "reason": "patient_deceased"}],
      ["update_patient", {"patient_id": "A-117", "status": "deceased"}]],
     lambda s, i: _final(s, i, changes={
         "appointments": {"AP-3005": {"status": "cancelled", "cancel_reason": "patient_deceased"},
                          "AP-3006": {"status": "cancelled", "cancel_reason": "patient_deceased"}},
         "prescriptions": {"RX-902": {"status": "stopped", "stopped": _VT, "stop_reason": "patient_deceased"}},
         "patients": {"A-117": {"status": "deceased"}}}),
     "bcdg", ref_errors=[2], mentions=["Grace Liu", "Maple"], seq=[[3, 7], [4, 7], [5, 7]])

_add("M03", "vet", _vet(),
     "Dr Mensah is off sick on Friday 13 November. Move each of his appointments still booked that day to Dr Sato, "
     "at the same time. Where that isn't possible, cancel the appointment instead, with the reason that fits, so "
     "reception can phone the owner. Anything already cancelled stays as it is, and nothing else should change.",
     [["get_clinic_policy", {}], ["list_appointments", {"date": "2026-11-13", "vet_id": "V-02"}],
      ["list_appointments", {"date": "2026-11-13", "vet_id": "V-05"}],
      ["reschedule_appointment", {"appointment_id": "AP-3030", "date": "2026-11-13", "time": "09:00", "vet_id": "V-05"}],
      ["reschedule_appointment", {"appointment_id": "AP-3031", "date": "2026-11-13", "time": "09:30", "vet_id": "V-05"}],
      ["reschedule_appointment", {"appointment_id": "AP-3032", "date": "2026-11-13", "time": "10:00", "vet_id": "V-05"}],
      ["cancel_appointment", {"appointment_id": "AP-3032", "reason": "vet_unavailable"}],
      ["reschedule_appointment", {"appointment_id": "AP-3033", "date": "2026-11-13", "time": "11:00", "vet_id": "V-05"}],
      ["list_appointments", {"date": "2026-11-13", "vet_id": "V-05"}]],
     [["reschedule_appointment", {"appointment_id": "AP-3030", "date": "2026-11-13", "time": "09:00", "vet_id": "V-05"}],
      ["reschedule_appointment", {"appointment_id": "AP-3031", "date": "2026-11-13", "time": "09:30", "vet_id": "V-05"}],
      ["reschedule_appointment", {"appointment_id": "AP-3032", "date": "2026-11-13", "time": "10:15", "vet_id": "V-05"}],
      ["reschedule_appointment", {"appointment_id": "AP-3033", "date": "2026-11-13", "time": "11:00", "vet_id": "V-05"}]],
     lambda s, i: _final(s, i, changes={"appointments": {
         "AP-3030": {"vet": "V-05"}, "AP-3031": {"vet": "V-05"}, "AP-3033": {"vet": "V-05"},
         "AP-3032": {"status": "cancelled", "cancel_reason": "vet_unavailable"}}}),
     "abdf", ref_errors=[5], mentions=["Mensah", "Sato"])

_add("M04", "vet", _vet(),
     "Pepper the spaniel has lost weight on her diet: the nurse weighed her this morning at 14,800 g, body "
     "condition 5. Log that, and replace her current amoxicillin prescription with one at the formulary dose for "
     "her new weight - same prescribing vet, same number of doses a day and the same food instruction as before, "
     "for 7 days. Record the old one as stopped for the dose change. Nothing else should change.",
     [["get_clinic_policy", {}], ["find_patients", {"query": "Pepper"}], ["get_patient", {"patient_id": "A-106"}],
      ["record_weight", {"patient_id": "A-106", "date": "2026-11-09", "weight_grams": 14800, "body_condition": 5}],
      ["get_dosing", {"drug": "amoxicillin", "species": "dog", "weight_grams": 14800}],
      ["prescribe_medication", {"patient_id": "A-106", "drug": "amoxicillin", "dose_mg": 296, "times_per_day": 2,
                                "days": 7, "vet_id": "V-01", "with_food": True}],
      ["stop_prescription", {"prescription_id": "RX-901", "reason": "dose_change"}],
      ["prescribe_medication", {"patient_id": "A-106", "drug": "amoxicillin", "dose_mg": 296, "times_per_day": 2,
                                "days": 7, "vet_id": "V-01", "with_food": True}],
      ["get_patient", {"patient_id": "A-106"}]],
     [["stop_prescription", {"prescription_id": "RX-901", "reason": "dose_change"}],
      ["prescribe_medication", {"patient_id": "A-112", "drug": "amoxicillin", "dose_mg": 296, "times_per_day": 2,
                                "days": 7, "vet_id": "V-01", "with_food": True}],
      ["record_weight", {"patient_id": "A-106", "date": "2026-11-09", "weight_grams": 14800, "body_condition": 5}],
      ["prescribe_medication", {"patient_id": "A-106", "drug": "amoxicillin", "dose_mg": 300, "times_per_day": 2,
                                "days": 7, "vet_id": "V-01", "with_food": True}]],
     lambda s, i: _final(s, i, changes={
         "prescriptions": {"RX-901": {"status": "stopped", "stopped": _VT, "stop_reason": "dose_change"}}},
         new={"weights": [_wt("A-106", _VT, 14800, 5)],
              "prescriptions": [_rx("A-106", "amoxicillin", 296, 2, 7, "V-01", True)]}),
     "bcef", ref_errors=[5], mentions=["Pepper"])

_M05_BATCH = {"dhpp": "DH-7781", "lepto": "LP-2093", "kennel_cough": "KC-4410", "rabies": "RB-5520",
              "fvrcp": "FV-1180", "felv": "FL-3012"}
_add("M05", "vet", _vet(),
     "Ines Okafor brought all of her animals in today and Dr Sato vaccinated them. For each animal, record every "
     "vaccine on the clinic schedule for its species that is due (next due date on or before today) or that it "
     "has never had - except a vaccine the animal is still too young for, which waits. All were injected under "
     "the skin. Batches used today: distemper combination DH-7781, leptospirosis LP-2093, kennel cough KC-4410, "
     "rabies RB-5520, cat flu and enteritis combination FV-1180, feline leukaemia FL-3012. Nothing else should change.",
     [["get_clinic_policy", {}], ["find_patients", {"query": "Okafor"}],
      ["get_patient", {"patient_id": "A-102"}], ["get_patient", {"patient_id": "A-111"}],
      ["get_patient", {"patient_id": "A-115"}],
      ["record_vaccinations", {"patient_id": "A-102", "date_given": "2026-11-09", "vet_id": "V-05",
                               "vaccines": [{"vaccine": "felv", "batch_number": "FL-3012"}]}],
      ["record_vaccinations", {"patient_id": "A-115", "date_given": "2026-11-09", "vet_id": "V-05", "vaccines": [
          {"vaccine": "lepto", "batch_number": "LP-2093"}, {"vaccine": "kennel_cough", "batch_number": "KC-4410"},
          {"vaccine": "rabies", "batch_number": "RB-5520"}]}],
      ["record_vaccinations", {"patient_id": "A-111", "date_given": "2026-11-09", "vet_id": "V-05", "vaccines": [
          {"vaccine": "fvrcp", "batch_number": "FV-1180"}, {"vaccine": "felv", "batch_number": "FL-3012"},
          {"vaccine": "rabies", "batch_number": "RB-5520"}]}],
      ["record_vaccinations", {"patient_id": "A-111", "date_given": "2026-11-09", "vet_id": "V-05", "vaccines": [
          {"vaccine": "fvrcp", "batch_number": "FV-1180"}, {"vaccine": "felv", "batch_number": "FL-3012"}]}],
      ["get_patient", {"patient_id": "A-111"}]],
     [["record_vaccinations", {"patient_id": "A-115", "date_given": "2026-11-09", "vet_id": "V-05", "vaccines": [
         {"vaccine": "dhpp", "batch_number": "DH-7781"}, {"vaccine": "lepto", "batch_number": "LP-2093"},
         {"vaccine": "kennel_cough", "batch_number": "KC-4410"}, {"vaccine": "rabies", "batch_number": "RB-5520"}]}],
      ["record_vaccinations", {"patient_id": "A-111", "date_given": "2026-11-09", "vet_id": "V-05", "vaccines": [
          {"vaccine": "fvrcp", "batch_number": "FV-1180"}, {"vaccine": "felv", "batch_number": "FL-3012"}]}]],
     lambda s, i: _final(s, i, new={"vaccinations": [
         _vx("A-102", "felv", _VT, "V-05", "FL-3012"),
         _vx("A-115", "lepto", _VT, "V-05", "LP-2093"), _vx("A-115", "kennel_cough", _VT, "V-05", "KC-4410"),
         _vx("A-115", "rabies", _VT, "V-05", "RB-5520"),
         _vx("A-111", "fvrcp", _VT, "V-05", "FV-1180", booster=1095), _vx("A-111", "felv", _VT, "V-05", "FL-3012")]}),
     "abdf", ref_errors=[7], mentions=["Ines Okafor", "Sato"],
     creates=list(_M05_BATCH.values()))

_add("M06", "vet", _vet(),
     "Ziggy the ferret is booked in for surgery with Dr Hart on 12 November. His owner has asked to move it to "
     "Saturday 21 November at the same time; if that time can't work, use the earliest start that does work that "
     "day, still with Dr Hart. For what it's worth, the nurse weighed him this morning at 1,180 g, body condition "
     "5 - it isn't on his record yet. Nothing else should change.",
     [["get_clinic_policy", {}], ["find_patients", {"query": "Ziggy"}],
      ["list_appointments", {"patient_id": "A-109"}],
      ["list_appointments", {"date": "2026-11-21", "vet_id": "V-03"}],
      ["reschedule_appointment", {"appointment_id": "AP-3011", "date": "2026-11-21", "time": "09:00"}],
      ["reschedule_appointment", {"appointment_id": "AP-3011", "date": "2026-11-21", "time": "10:40"}],
      ["record_weight", {"patient_id": "A-109", "date": "2026-11-09", "weight_grams": 1180, "body_condition": 5}],
      ["reschedule_appointment", {"appointment_id": "AP-3011", "date": "2026-11-21", "time": "10:40"}],
      ["list_appointments", {"patient_id": "A-109"}]],
     [["reschedule_appointment", {"appointment_id": "AP-3011", "date": "2026-11-21", "time": "09:00"}],
      ["record_weight", {"patient_id": "A-109", "date": "2026-11-09", "weight_grams": 1180, "body_condition": 5}],
      ["reschedule_appointment", {"appointment_id": "AP-3011", "date": "2026-11-21", "time": "10:45"}]],
     lambda s, i: _final(s, i, changes={"appointments": {"AP-3011": {"date": "2026-11-21", "time": "10:40"}}},
                         new={"weights": [_wt("A-109", _VT, 1180, 5)]}),
     "abcef", ref_errors=[4, 5], mentions=["Ziggy", "Hart"])

# ---- calibration C1 (single), C6 (multi)
_add("C1", "vet", _vet(),
     "Please book Saffron the cockatiel (A-114) in with Dr Hart (V-03) for a nail trim on Thursday 19 November "
     "2026. The clinic opens at 08:00 that day and the owner wants the slot that starts 1 h 20 min after opening. "
     "She would like a text message two days before and an email reminder a week before.",
     [["schedule_appointment", {"patient_id": "A-114", "vet_id": "V-03", "date": "2026-11-19", "time": "09:20",
                                "type": "nail_trim", "reminders": [{"channel": "sms", "days_before": 2},
                                                                   {"channel": "email", "days_before": 7}]}]],
     [["schedule_appointment", {"patient_id": "A-114", "vet_id": "V-03", "date": "2026-11-19", "time": "09:20",
                                "type": "nail_trim", "reminders": [{"channel": "sms", "days_before": 2}]}]],
     lambda s, i: _final(s, i, new={"appointments": [_appt(
         "A-114", "V-03", "2026-11-19", "09:20", "nail_trim",
         reminders=[{"channel": "sms", "days_before": 2}, {"channel": "email", "days_before": 7}])]}),
     "OUN", mentions=["Saffron", "Hart"])

_add("C6", "vet", _vet(),
     "The clinic will be closed on Saturday 14 November for a boiler repair. Move every appointment booked that "
     "day to the next day its vet works, at the same time and with the same vet. If the vet or the animal is "
     "already busy at that time on that day, cancel the appointment instead, with the reason that fits. Nothing "
     "else should change.",
     [["get_clinic_policy", {}], ["list_appointments", {"date": "2026-11-14"}],
      ["reschedule_appointment", {"appointment_id": "AP-3040", "date": "2026-11-17", "time": "09:00"}],
      ["reschedule_appointment", {"appointment_id": "AP-3046", "date": "2026-11-17", "time": "10:00"}],
      ["cancel_appointment", {"appointment_id": "AP-3046", "reason": "clinic_closure"}],
      ["reschedule_appointment", {"appointment_id": "AP-3045", "date": "2026-11-16", "time": "11:00"}],
      ["cancel_appointment", {"appointment_id": "AP-3045", "reason": "clinic_closure"}],
      ["reschedule_appointment", {"appointment_id": "AP-3051", "date": "2026-11-17", "time": "12:00"}],
      ["list_appointments", {"date": "2026-11-14"}]],
     [["reschedule_appointment", {"appointment_id": "AP-3040", "date": "2026-11-17", "time": "09:00"}],
      ["reschedule_appointment", {"appointment_id": "AP-3045", "date": "2026-11-18", "time": "11:00"}],
      ["reschedule_appointment", {"appointment_id": "AP-3046", "date": "2026-11-18", "time": "10:00"}],
      ["reschedule_appointment", {"appointment_id": "AP-3051", "date": "2026-11-17", "time": "12:00"}]],
     lambda s, i: _final(s, i, changes={"appointments": {
         "AP-3040": {"date": "2026-11-17"}, "AP-3051": {"date": "2026-11-17"},
         "AP-3045": {"status": "cancelled", "cancel_reason": "clinic_closure"},
         "AP-3046": {"status": "cancelled", "cancel_reason": "clinic_closure"}}}),
     "abdf", ref_errors=[3, 5])


# --------------------------------------------------------------------------
# 2. warehouse pick/pack: orders, bins, stock lots (FEFO), carriers and cut-offs, shipments
# --------------------------------------------------------------------------
_ZONES = ["ambient", "bulk", "chilled", "hazmat"]
_SERVICES = ["economy", "standard", "express", "next_day"]
_CARRIERS = ["swiftpost", "northline", "parceldirect", "courierblue"]
_ORDER_STATUS = ["new", "allocated", "on_hold", "shipped", "cancelled"]
_ADJ_REASONS = ["cycle_count", "damaged", "found"]
_Q_REASONS = ["temperature_breach", "damaged_packaging", "recall", "contamination"]
_HOLD_REASONS = ["customer_request", "address_check", "stock_issue", "carrier_issue"]
_OPEN_ORDER = ("new", "allocated", "on_hold")


class WarehouseEnv(_Env):
    TOOLS = [
        _fn("get_warehouse_policy", "Today's date, the current time, the carriers (services, today's cut-off, "
            "parcel limits, hazardous goods), minimum shelf life per storage type and the warehouse rules that the "
            "write tools enforce."),
        _fn("list_orders", "Orders, filtered, by order id; 8 per page.",
            {"status": _s("Only orders in this status.", _ORDER_STATUS), "carrier": _s("Only this carrier.", _CARRIERS),
             "ship_by": _s("Only orders with this ship-by date, YYYY-MM-DD."), "page": _PAGE}),
        _fn("get_order", "One order: lines (with product, storage type, unit weight), goods weight, allocations, "
            "carrier, service, hold details.", {"order_id": _s("Order id, e.g. OR-5104.")}, ["order_id"]),
        _fn("find_stock", "Stock lots of one product, by expiry date: bin, quantity, quantity allocated to orders "
            "(and which), free quantity, expiry and status.",
            {"sku": _s("Product id, e.g. SK-120."),
             "include_unavailable": _b("Also list quarantined and empty lots (default false).")}, ["sku"]),
        _fn("list_bins", "Bin locations, filtered, by bin id; 10 per page: zone, status, capacity in units, units "
            "stored, free capacity and the products in it.",
            {"zone": _s("Only this zone.", _ZONES), "aisle": _s("Only this aisle letter, e.g. A."), "page": _PAGE}),
        _fn("allocate_stock", "Allocate stock lots to an order's lines (adds to any existing allocation). When "
            "every line is fully allocated the order becomes 'allocated'.",
            {"order_id": _s("Order id."),
             "picks": _a("Lots and quantities to allocate.",
                         _o({"lot_id": _s("Lot id, e.g. LT-1201."), "qty": _i("Units from that lot.", 1, 10000)},
                            ["lot_id", "qty"]), 1, 10)},
            ["order_id", "picks"]),
        _fn("release_allocation", "Remove every allocation from an order that has not shipped; it goes back to 'new'.",
            {"order_id": _s("Order id.")}, ["order_id"]),
        _fn("move_stock", "Move units of a lot to another bin. Moving the whole lot keeps its lot id (and its "
            "allocations); moving part of it creates a new lot in the destination bin.",
            {"lot_id": _s("Lot id."), "to_bin": _s("Destination bin id, e.g. BN-A03."),
             "qty": _i("Units to move.", 1, 10000)}, ["lot_id", "to_bin", "qty"]),
        _fn("adjust_stock", "Correct the quantity of a lot after a count, damage or a find.",
            {"lot_id": _s("Lot id."), "counted_qty": _i("The correct quantity now in the lot.", 0, 10000),
             "reason": _s("Why the quantity changes.", _ADJ_REASONS)}, ["lot_id", "counted_qty", "reason"]),
        _fn("quarantine_lot", "Block a whole lot from being picked.",
            {"lot_id": _s("Lot id."), "reason": _s("Why.", _Q_REASONS)}, ["lot_id", "reason"]),
        _fn("create_shipment", "Book the shipment of a fully allocated order with its carrier: the allocated stock "
            "leaves the warehouse and the order becomes 'shipped'.",
            {"order_id": _s("Order id."), "carrier": _s("Carrier.", _CARRIERS), "service": _s("Service.", _SERVICES),
             "ship_date": _s("Date the parcels leave, YYYY-MM-DD."),
             "parcels": _a("The parcels (dimensions are stored longest side first).", _o({"weight_g": _i("Gross weight in grams.", 1, 100000),
                                                "length_cm": _i("Length in cm.", 1, 300),
                                                "width_cm": _i("Width in cm.", 1, 300),
                                                "height_cm": _i("Height in cm.", 1, 300)},
                                               ["weight_g", "length_cm", "width_cm", "height_cm"]), 1, 10)},
            ["order_id", "carrier", "service", "ship_date", "parcels"]),
        _fn("change_carrier", "Change the carrier and service an unshipped order is booked with.",
            {"order_id": _s("Order id."), "carrier": _s("New carrier.", _CARRIERS),
             "service": _s("New service.", _SERVICES)}, ["order_id", "carrier", "service"]),
        _fn("set_order_hold", "Put an order on hold (on_hold=true, with a reason) or release it from hold "
            "(on_hold=false; it returns to the status it had before).",
            {"order_id": _s("Order id."), "on_hold": _b("true to hold, false to release."),
             "reason": _s("Why it is held (required when holding).", _HOLD_REASONS),
             "review_on": _s("Optional date to review the hold, YYYY-MM-DD.")}, ["order_id", "on_hold"]),
        _fn("split_order", "Move some quantities of a new, unallocated order into a new backorder (same customer, "
            "carrier, service, priority and ship-by date).",
            {"order_id": _s("Order id."),
             "lines": _a("Quantities to move to the backorder.",
                         _o({"sku": _s("Product id."), "qty": _i("Units to move.", 1, 10000)}, ["sku", "qty"]), 1, 10)},
            ["order_id", "lines"]),
    ]

    def _order(self, oid):
        return self._get("orders", oid, "order")

    def _lot(self, lid):
        return self._get("lots", lid, "lot")

    def _bin(self, bid):
        return self._get("bins", bid, "bin")

    def _allocated(self, lid):
        return sum(x["qty"] for o in self.state["orders"].values() if o["status"] in _OPEN_ORDER
                   for x in o["alloc"] if x["lot"] == lid)

    def _alloc_orders(self, lid):
        return sorted(k for k, o in self.state["orders"].items() if o["status"] in _OPEN_ORDER
                      and any(x["lot"] == lid for x in o["alloc"]))

    def _free(self, lid):
        return self.state["lots"][lid]["qty"] - self._allocated(lid)

    def _zone_ok(self, sku, zone):
        st = self.state["skus"][sku]["storage"]
        return zone in (("ambient", "bulk") if st == "ambient" else (st,))

    def _bin_units(self, bid):
        return sum(l["qty"] for l in self.state["lots"].values() if l["bin"] == bid)

    def _bin_skus(self, bid):
        return sorted(set(l["sku"] for l in self.state["lots"].values() if l["bin"] == bid and l["qty"] > 0))

    def _min_expiry(self, order, sku):
        return _plus_days(order["ship_by"], self.state["policy"]["min_shelf_days"][self.state["skus"][sku]["storage"]])

    def _eligible(self, lid, order, sku):
        l = self.state["lots"][lid]
        return l["sku"] == sku and l["status"] == "available" and self.state["bins"][l["bin"]]["status"] == "active" \
            and l["expiry"] >= self._min_expiry(order, sku)

    def _goods_g(self, o):
        return sum(ln["qty"] * self.state["skus"][ln["sku"]]["unit_g"] for ln in o["lines"])

    def _hazmat(self, o):
        return any(self.state["skus"][ln["sku"]]["storage"] == "hazmat" for ln in o["lines"])

    def _order_row(self, k):
        o = self.state["orders"][k]
        return {"order_id": k, "customer": o["customer"], "priority": o["priority"], "ship_by": o["ship_by"],
                "carrier": o["carrier"], "service": o["service"], "status": o["status"],
                "lines": len(o["lines"]), "units": sum(ln["qty"] for ln in o["lines"])}

    def _t_get_warehouse_policy(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "weekday": _DAYNAMES[self._today().weekday()], "now": p["now"],
                "carriers": [dict(carrier=k, **c) for k, c in sorted(p["carriers"].items())],
                "min_shelf_days": p["min_shelf_days"], "services_allowed_by_priority": p["priority_services"],
                "rules": [
                    "Allocation follows FEFO: for each product, stock must come from the eligible lot with the "
                    "earliest expiry date (ties: lower lot id) before any later-expiring lot is used. Eligible lots "
                    "are available (not quarantined), sit in an active bin, and expire no earlier than the order's "
                    "ship-by date plus min_shelf_days for the product's storage type. Ineligible lots are skipped.",
                    "Only free quantity (lot quantity minus what open orders have allocated) can be allocated or "
                    "moved; a lot's quantity cannot be adjusted below what is allocated from it, and a lot with "
                    "allocations cannot be quarantined - release the orders' allocations first.",
                    "Bins hold one product only. Ambient products go to ambient or bulk bins, chilled products to "
                    "chilled bins, hazmat products to hazmat bins. Blocked bins cannot receive stock or be picked "
                    "from, but stock may be moved out of them. A bin's units may not exceed its capacity.",
                    "A shipment needs a fully allocated order that is not on hold, booked with the same carrier "
                    "and service as the order (use change_carrier to change them). Ship dates must be working days "
                    "(Monday-Friday), not in the past and not later than the ship-by date; shipping today is only "
                    "possible before the carrier's cut-off time. Orders with hazmat products need a carrier that "
                    "takes hazardous goods. No parcel may exceed the carrier's max_parcel_g, and the parcels "
                    "together must weigh at least the goods weight.",
                    "The service must be one the carrier offers and one the order's priority allows.",
                    "Orders can only be split while new and unallocated; at least one unit must stay on the "
                    "original order.",
                ]}

    def _t_list_orders(self, a):
        if "ship_by" in a:
            _pdate(a["ship_by"], "ship_by")
        rows = [k for k, o in sorted(self.state["orders"].items())
                if a.get("status", o["status"]) == o["status"] and a.get("carrier", o["carrier"]) == o["carrier"]
                and a.get("ship_by", o["ship_by"]) == o["ship_by"]]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["orders"] = [self._order_row(k) for k in rows]
        return meta

    def _t_get_order(self, a):
        k = a["order_id"]
        o = self._order(k)
        row = dict(order_id=k, **o)
        row["lines"] = [{"sku": ln["sku"], "name": self.state["skus"][ln["sku"]]["name"], "qty": ln["qty"],
                         "storage": self.state["skus"][ln["sku"]]["storage"],
                         "unit_g": self.state["skus"][ln["sku"]]["unit_g"],
                         "allocated": sum(x["qty"] for x in o["alloc"] if x["sku"] == ln["sku"])} for ln in o["lines"]]
        row["goods_weight_g"] = self._goods_g(o)
        row["contains_hazmat"] = self._hazmat(o)
        return row

    def _t_find_stock(self, a):
        sku = a["sku"]
        s = self._get("skus", sku, "product")
        rows = []
        for _, k in sorted((l["expiry"], k) for k, l in self.state["lots"].items() if l["sku"] == sku):
            l = self.state["lots"][k]
            if not a.get("include_unavailable", False) and (l["status"] != "available" or l["qty"] == 0):
                continue
            allocs = [{"order_id": ok, "qty": x["qty"]} for ok in self._alloc_orders(k)
                      for x in self.state["orders"][ok]["alloc"] if x["lot"] == k]
            rows.append({"lot_id": k, "bin": l["bin"], "bin_status": self.state["bins"][l["bin"]]["status"],
                         "qty": l["qty"], "allocated": self._allocated(k), "free": self._free(k),
                         "allocations": allocs, "expiry": l["expiry"], "status": l["status"]})
        return {"sku": sku, "name": s["name"], "storage": s["storage"], "unit_g": s["unit_g"], "lots": rows}

    def _t_list_bins(self, a):
        rows = [k for k, b in sorted(self.state["bins"].items())
                if a.get("zone", b["zone"]) == b["zone"] and ("aisle" not in a or b["aisle"] == a["aisle"].strip().upper())]
        meta, rows = _paged(rows, a.get("page", 1), 10)
        meta["bins"] = [{"bin_id": k, "zone": self.state["bins"][k]["zone"], "aisle": self.state["bins"][k]["aisle"],
                         "status": self.state["bins"][k]["status"], "capacity": self.state["bins"][k]["capacity"],
                         "units": self._bin_units(k), "free_capacity": self.state["bins"][k]["capacity"] - self._bin_units(k),
                         "products": self._bin_skus(k),
                         "lots": sorted(lk for lk, l in self.state["lots"].items() if l["bin"] == k and l["qty"] > 0)}
                        for k in rows]
        return meta

    def _t_allocate_stock(self, a):
        k = a["order_id"]
        o = self._order(k)
        if o["status"] == "on_hold":
            raise _Rule("order %s is on hold" % k)
        if o["status"] != "new":
            raise _Rule("order %s is %s; only new orders can be allocated" % (k, o["status"]))
        need = dict((ln["sku"], ln["qty"] - sum(x["qty"] for x in o["alloc"] if x["sku"] == ln["sku"]))
                    for ln in o["lines"])
        for p in a["picks"]:
            lid = p["lot_id"]
            l = self._lot(lid)
            if l["sku"] not in need:
                raise _Rule("lot %s holds %s, which is not on order %s" % (lid, l["sku"], k))
            if l["status"] != "available":
                raise _Rule("lot %s is %s" % (lid, l["status"]))
            if self.state["bins"][l["bin"]]["status"] != "active":
                raise _Rule("lot %s is in bin %s, which is blocked" % (lid, l["bin"]))
            if l["expiry"] < self._min_expiry(o, l["sku"]):
                raise _Rule("lot %s expires %s; for ship-by %s a %s product must not expire before %s" % (
                    lid, l["expiry"], o["ship_by"], self.state["skus"][l["sku"]]["storage"], self._min_expiry(o, l["sku"])))
            if p["qty"] > self._free(lid):
                raise _Rule("lot %s has only %d free units" % (lid, self._free(lid)))
            if p["qty"] > need[l["sku"]]:
                raise _Rule("order %s needs only %d more units of %s" % (k, need[l["sku"]], l["sku"]))
            need[l["sku"]] -= p["qty"]
            for x in o["alloc"]:
                if x["lot"] == lid:
                    x["qty"] += p["qty"]
                    break
            else:
                o["alloc"].append({"sku": l["sku"], "lot": lid, "qty": p["qty"]})
        # FEFO, checked on the result (independent of the order of the picks)
        for x in o["alloc"]:
            mine = self.state["lots"][x["lot"]]
            for k2 in sorted(self.state["lots"]):
                if k2 != x["lot"] and self._eligible(k2, o, x["sku"]) and self._free(k2) > 0 and \
                        (self.state["lots"][k2]["expiry"], k2) < (mine["expiry"], x["lot"]):
                    raise _Rule("FEFO: lot %s (expires %s) has %d free units and must be used before lot %s "
                                "(expires %s)" % (k2, self.state["lots"][k2]["expiry"], self._free(k2), x["lot"],
                                                  mine["expiry"]))
        o["alloc"] = sorted(o["alloc"], key=lambda x: (x["sku"], x["lot"]))
        if all(v == 0 for v in need.values()):
            o["status"] = "allocated"
        return {"ok": True, "order": self._t_get_order({"order_id": k})}

    def _t_release_allocation(self, a):
        k = a["order_id"]
        o = self._order(k)
        if o["status"] == "on_hold":
            raise _Rule("order %s is on hold; release the hold first" % k)
        if o["status"] not in ("new", "allocated") or not o["alloc"]:
            raise _Rule("order %s (%s) has no allocation to release" % (k, o["status"]))
        o["alloc"], o["status"] = [], "new"
        return {"ok": True, "order": self._t_get_order({"order_id": k})}

    def _t_move_stock(self, a):
        lid, bid, q = a["lot_id"], a["to_bin"], a["qty"]
        l = self._lot(lid)
        b = self._bin(bid)
        if l["status"] != "available":
            raise _Rule("lot %s is %s and cannot be moved" % (lid, l["status"]))
        if l["bin"] == bid:
            raise _Rule("lot %s is already in %s" % (lid, bid))
        if b["status"] != "active":
            raise _Rule("bin %s is blocked" % bid)
        if not self._zone_ok(l["sku"], b["zone"]):
            raise _Rule("%s is a %s product and cannot go to a %s bin" % (
                l["sku"], self.state["skus"][l["sku"]]["storage"], b["zone"]))
        other = [s for s in self._bin_skus(bid) if s != l["sku"]]
        if other:
            raise _Rule("bin %s already holds %s; bins hold one product only" % (bid, ", ".join(other)))
        if q > l["qty"]:
            raise _Rule("lot %s has only %d units" % (lid, l["qty"]))
        if self._bin_units(bid) + q > b["capacity"]:
            raise _Rule("bin %s has room for %d more units" % (bid, b["capacity"] - self._bin_units(bid)))
        if q == l["qty"]:
            l["bin"] = bid
            return {"ok": True, "lot_id": lid, "bin": bid, "qty": q}
        if q > self._free(lid):
            raise _Rule("only %d units of lot %s are free; allocated units can only move with the whole lot" % (
                self._free(lid), lid))
        l["qty"] -= q
        n = self._new_id("next_lot", "LT")
        self.state["lots"][n] = {"sku": l["sku"], "bin": bid, "qty": q, "expiry": l["expiry"], "status": "available",
                                 "q_reason": ""}
        return {"ok": True, "lot_id": n, "split_from": lid, "bin": bid, "qty": q}

    def _t_adjust_stock(self, a):
        lid = a["lot_id"]
        l = self._lot(lid)
        if a["counted_qty"] == l["qty"]:
            raise _Rule("lot %s already shows %d units" % (lid, l["qty"]))
        if a["counted_qty"] < self._allocated(lid):
            raise _Rule("%d units of lot %s are allocated (%s); release those allocations before adjusting below "
                        "that" % (self._allocated(lid), lid, ", ".join(self._alloc_orders(lid))))
        if self._bin_units(l["bin"]) - l["qty"] + a["counted_qty"] > self.state["bins"][l["bin"]]["capacity"]:
            raise _Rule("bin %s cannot hold that many units" % l["bin"])
        n = self._new_id("next_adj", "ADJ")
        self.state["adjustments"][n] = {"lot": lid, "from_qty": l["qty"], "to_qty": a["counted_qty"],
                                        "reason": a["reason"], "date": self.state["today"]}
        l["qty"] = a["counted_qty"]
        return {"ok": True, "adjustment_id": n, "lot_id": lid, "qty": l["qty"]}

    def _t_quarantine_lot(self, a):
        lid = a["lot_id"]
        l = self._lot(lid)
        if l["status"] != "available":
            raise _Rule("lot %s is already %s" % (lid, l["status"]))
        if self._allocated(lid):
            raise _Rule("lot %s has %d units allocated to %s; release those allocations first" % (
                lid, self._allocated(lid), ", ".join(self._alloc_orders(lid))))
        l["status"], l["q_reason"] = "quarantine", a["reason"]
        return {"ok": True, "lot_id": lid, "status": "quarantine"}

    def _service_ok(self, o, carrier, service):
        c = self.state["policy"]["carriers"][carrier]
        if service not in c["services"]:
            raise _Rule("%s does not offer %s (offers: %s)" % (carrier, service, ", ".join(c["services"])))
        allowed = self.state["policy"]["priority_services"][o["priority"]]
        if service not in allowed:
            raise _Rule("a %s order must ship %s" % (o["priority"], " or ".join(allowed)))
        if self._hazmat(o) and not c["hazmat"]:
            raise _Rule("%s does not take hazardous goods" % carrier)

    def _t_create_shipment(self, a):
        k = a["order_id"]
        o = self._order(k)
        if o["status"] != "allocated":
            raise _Rule("order %s is %s; only fully allocated orders that are not on hold can ship" % (k, o["status"]))
        if (a["carrier"], a["service"]) != (o["carrier"], o["service"]):
            raise _Rule("order %s is booked with %s %s; change the carrier first to ship otherwise" % (
                k, o["carrier"], o["service"]))
        self._service_ok(o, a["carrier"], a["service"])
        c = self.state["policy"]["carriers"][a["carrier"]]
        d = _pdate(a["ship_date"], "ship_date")
        if d < self._today():
            raise _Rule("ship_date %s is in the past" % a["ship_date"])
        if d.weekday() >= 5:
            raise _Rule("%s is a %s; carriers collect Monday to Friday" % (a["ship_date"], _DAYNAMES[d.weekday()]))
        if a["ship_date"] == self.state["today"] and _ptime(self.state["policy"]["now"]) >= _ptime(c["cutoff"]):
            raise _Rule("today's cut-off for %s was %s (it is now %s)" % (a["carrier"], c["cutoff"],
                                                                          self.state["policy"]["now"]))
        if a["ship_date"] > o["ship_by"]:
            raise _Rule("order %s must ship by %s" % (k, o["ship_by"]))
        total = 0
        for p in a["parcels"]:
            if p["weight_g"] > c["max_parcel_g"]:
                raise _Rule("%s takes parcels up to %d g" % (a["carrier"], c["max_parcel_g"]))
            total += p["weight_g"]
        if total < self._goods_g(o):
            raise _Rule("the parcels weigh %d g in total, less than the goods (%d g)" % (total, self._goods_g(o)))
        parcels = []   # a box's dimensions are recorded longest side first (length >= width >= height)
        for p in a["parcels"]:
            d = sorted([p["length_cm"], p["width_cm"], p["height_cm"]], reverse=True)
            parcels.append({"weight_g": p["weight_g"], "length_cm": d[0], "width_cm": d[1], "height_cm": d[2]})
        for x in o["alloc"]:
            self.state["lots"][x["lot"]]["qty"] -= x["qty"]
        o["status"] = "shipped"
        n = self._new_id("next_shipment", "SH")
        self.state["shipments"][n] = {
            "order": k, "carrier": a["carrier"], "service": a["service"], "ship_date": a["ship_date"],
            "parcels": sorted(parcels, key=lambda p: (p["weight_g"], p["length_cm"], p["width_cm"], p["height_cm"]))}
        return {"ok": True, "shipment_id": n, "shipment": self.state["shipments"][n]}

    def _t_change_carrier(self, a):
        k = a["order_id"]
        o = self._order(k)
        if o["status"] not in _OPEN_ORDER:
            raise _Rule("order %s is %s" % (k, o["status"]))
        if (a["carrier"], a["service"]) == (o["carrier"], o["service"]):
            raise _Rule("order %s is already booked with %s %s" % (k, o["carrier"], o["service"]))
        self._service_ok(o, a["carrier"], a["service"])
        o["carrier"], o["service"] = a["carrier"], a["service"]
        return {"ok": True, "order": self._order_row(k)}

    def _t_set_order_hold(self, a):
        k = a["order_id"]
        o = self._order(k)
        if a["on_hold"]:
            if o["status"] not in ("new", "allocated"):
                raise _Rule("order %s is %s and cannot be held" % (k, o["status"]))
            if "reason" not in a:
                raise _Rule("a reason is required to put an order on hold")
            if "review_on" in a and _pdate(a["review_on"], "review_on") < self._today():
                raise _Rule("review_on is in the past")
            o["held_from"], o["status"] = o["status"], "on_hold"
            o["hold_reason"], o["hold_until"] = a["reason"], a.get("review_on", "")
        else:
            if o["status"] != "on_hold":
                raise _Rule("order %s is not on hold" % k)
            if "reason" in a or "review_on" in a:
                raise _Rule("reason and review_on are only used when putting an order on hold")
            o["status"], o["held_from"], o["hold_reason"], o["hold_until"] = o["held_from"], "", "", ""
        return {"ok": True, "order": self._order_row(k)}

    def _t_split_order(self, a):
        k = a["order_id"]
        o = self._order(k)
        if o["status"] != "new" or o["alloc"]:
            raise _Rule("order %s is %s%s; only new, unallocated orders can be split" % (
                k, o["status"], " with allocations" if o["alloc"] else ""))
        qty = dict((ln["sku"], ln["qty"]) for ln in o["lines"])
        moved = {}
        for ln in a["lines"]:
            if ln["sku"] not in qty:
                raise _Rule("%s is not on order %s" % (ln["sku"][:20], k))
            if ln["sku"] in moved:
                raise _Rule("%s is listed twice" % ln["sku"])
            if ln["qty"] > qty[ln["sku"]]:
                raise _Rule("order %s has only %d units of %s" % (k, qty[ln["sku"]], ln["sku"]))
            moved[ln["sku"]] = ln["qty"]
        if sum(qty.values()) == sum(moved.values()):
            raise _Rule("at least one unit must stay on order %s" % k)
        o["lines"] = [{"sku": s, "qty": qty[s] - moved.get(s, 0)} for s in sorted(qty) if qty[s] - moved.get(s, 0) > 0]
        n = self._new_id("next_order", "OR")
        self.state["orders"][n] = {
            "customer": o["customer"], "priority": o["priority"], "ship_by": o["ship_by"], "carrier": o["carrier"],
            "service": o["service"], "status": "new", "lines": [{"sku": s, "qty": moved[s]} for s in sorted(moved)],
            "alloc": [], "hold_reason": "", "hold_until": "", "held_from": "", "backorder_of": k}
        return {"ok": True, "order": self._order_row(k), "backorder": self._order_row(n)}


def _wh():
    skus = dict((k, {"name": n, "storage": st, "unit_g": g}) for k, n, st, g in [
        ("SK-110", "Oat Milk 1L, case of 6", "ambient", 6400), ("SK-120", "Coffee Beans 1kg", "ambient", 1050),
        ("SK-130", "Greek Yogurt 500g, tray of 8", "chilled", 4300), ("SK-140", "Paper Towels 12-roll", "ambient", 2100),
        ("SK-150", "Olive Oil 750ml", "ambient", 1300), ("SK-160", "Hand Sanitiser 500ml", "hazmat", 560),
        ("SK-170", "Spray Paint Aerosol 400ml", "hazmat", 450), ("SK-180", "Fresh Pasta 400g", "chilled", 420),
        ("SK-190", "Dish Soap 1L", "ambient", 1100), ("SK-200", "Dog Food 12kg", "ambient", 12200)])
    bins = {}
    for n in range(1, 13):
        bins["BN-A%02d" % n] = {"zone": "ambient", "aisle": "A", "capacity": 120, "status": "active"}
    for n in range(1, 5):
        bins["BN-K%02d" % n] = {"zone": "bulk", "aisle": "K", "capacity": 600, "status": "active"}
    for n in range(1, 7):
        bins["BN-C%02d" % n] = {"zone": "chilled", "aisle": "C", "capacity": 80, "status": "active"}
    for n in range(1, 4):
        bins["BN-H%02d" % n] = {"zone": "hazmat", "aisle": "H", "capacity": 100, "status": "active"}
    bins["BN-A07"]["status"] = "blocked"
    lots = {}
    for k, sku, b, q, exp, status in [
        ("LT-1101", "SK-110", "BN-A01", 60, "2027-03-01", "available"),
        ("LT-1102", "SK-110", "BN-K01", 300, "2027-05-20", "available"),
        ("LT-1201", "SK-120", "BN-A02", 40, "2027-01-15", "available"),
        ("LT-1202", "SK-120", "BN-A07", 25, "2026-12-20", "available"),
        ("LT-1203", "SK-120", "BN-K02", 200, "2027-06-30", "available"),
        ("LT-1301", "SK-130", "BN-C01", 10, "2026-11-13", "available"),
        ("LT-1302", "SK-130", "BN-C01", 18, "2026-11-20", "available"),
        ("LT-1303", "SK-130", "BN-C02", 40, "2026-11-24", "available"),
        ("LT-1304", "SK-130", "BN-C03", 30, "2026-12-01", "available"),
        ("LT-1305", "SK-130", "BN-C01", 25, "2026-11-28", "available"),
        ("LT-1401", "SK-140", "BN-A03", 46, "2027-12-31", "available"),
        ("LT-1402", "SK-140", "BN-K03", 400, "2028-06-30", "available"),
        ("LT-1501", "SK-150", "BN-A04", 20, "2027-02-10", "available"),
        ("LT-1502", "SK-150", "BN-A05", 15, "2026-12-01", "available"),
        ("LT-1503", "SK-150", "BN-A08", 24, "2027-04-01", "available"),
        ("LT-1601", "SK-160", "BN-H01", 50, "2027-08-01", "available"),
        ("LT-1602", "SK-160", "BN-H01", 20, "2026-12-15", "available"),
        ("LT-1701", "SK-170", "BN-H03", 30, "2028-01-01", "available"),
        ("LT-1801", "SK-180", "BN-C04", 25, "2026-11-18", "available"),
        ("LT-1802", "SK-180", "BN-C05", 35, "2026-11-22", "available"),
        ("LT-1803", "SK-180", "BN-C05", 12, "2026-11-25", "quarantine"),
        ("LT-1901", "SK-190", "BN-A06", 80, "2027-09-09", "available"),
        ("LT-2001", "SK-200", "BN-K04", 30, "2027-04-04", "available"),
    ]:
        lots[k] = {"sku": sku, "bin": b, "qty": q, "expiry": exp, "status": status,
                   "q_reason": "damaged_packaging" if status == "quarantine" else ""}
    orders = {}
    for k, cust, prio, ship_by, carrier, service, status, lines, alloc in [
        ("OR-5104", "Harbour Cafe", "standard", "2026-11-12", "northline", "standard", "allocated",
         [("SK-110", 2), ("SK-120", 4)], [("SK-110", "LT-1101", 2), ("SK-120", "LT-1201", 4)]),
        ("OR-5110", "Green Grocer Co", "standard", "2026-11-11", "swiftpost", "standard", "new",
         [("SK-120", 12), ("SK-130", 30)], []),
        ("OR-5121", "Pine and Co", "express", "2026-11-13", "parceldirect", "express", "new", [("SK-190", 10)], []),
        ("OR-5130", "Bay Deli", "standard", "2026-11-16", "northline", "economy", "new",
         [("SK-150", 20), ("SK-160", 6), ("SK-190", 5)], []),
        ("OR-5140", "Harbour Cafe", "standard", "2026-11-17", "swiftpost", "standard", "new", [("SK-200", 2)], []),
        ("OR-5150", "Alder Foods", "standard", "2026-11-12", "parceldirect", "standard", "new",
         [("SK-120", 6), ("SK-150", 4)], []),
        ("OR-5161", "Harbour Cafe", "standard", "2026-11-10", "swiftpost", "standard", "allocated",
         [("SK-120", 3), ("SK-190", 2)], [("SK-120", "LT-1201", 3), ("SK-190", "LT-1901", 2)]),
        ("OR-5162", "Lumen Bakery", "express", "2026-11-10", "parceldirect", "express", "allocated",
         [("SK-150", 5)], [("SK-150", "LT-1501", 5)]),
        ("OR-5163", "Juniper Pets", "standard", "2026-11-10", "parceldirect", "standard", "allocated",
         [("SK-140", 4)], [("SK-140", "LT-1401", 4)]),
        ("OR-5164", "Lumen Bakery", "standard", "2026-11-10", "swiftpost", "standard", "on_hold",
         [("SK-190", 3)], [("SK-190", "LT-1901", 3)]),
        ("OR-5165", "Pine and Co", "standard", "2026-11-11", "parceldirect", "standard", "allocated",
         [("SK-110", 1)], [("SK-110", "LT-1101", 1)]),
        ("OR-5166", "Alder Foods", "standard", "2026-11-10", "northline", "standard", "new", [("SK-200", 1)], []),
        ("OR-5170", "Copper Kettle", "standard", "2026-11-13", "northline", "standard", "new",
         [("SK-180", 70), ("SK-190", 4)], []),
        ("OR-5175", "Studio Nine", "express", "2026-11-11", "swiftpost", "express", "allocated",
         [("SK-140", 2), ("SK-170", 6)], [("SK-140", "LT-1401", 2), ("SK-170", "LT-1701", 6)]),
        ("OR-5090", "Bay Deli", "standard", "2026-11-06", "northline", "standard", "shipped",
         [("SK-110", 5)], [("SK-110", "LT-1101", 5)]),
        ("OR-5091", "Green Grocer Co", "standard", "2026-11-09", "swiftpost", "standard", "cancelled",
         [("SK-130", 8)], []),
    ]:
        orders[k] = {"customer": cust, "priority": prio, "ship_by": ship_by, "carrier": carrier, "service": service,
                     "status": status, "lines": [{"sku": s, "qty": q} for s, q in lines],
                     "alloc": [{"sku": s, "lot": l, "qty": q} for s, l, q in alloc],
                     "hold_reason": "", "hold_until": "", "held_from": "", "backorder_of": ""}
    orders["OR-5164"].update({"hold_reason": "customer_request", "held_from": "allocated"})
    st = {"today": "2026-11-10",
          "policy": {"now": "14:20", "min_shelf_days": {"ambient": 30, "chilled": 5, "hazmat": 60},
                     "priority_services": {"standard": _SERVICES, "express": ["express", "next_day"],
                                           "next_day": ["next_day"]},
                     "carriers": {
                         "swiftpost": {"name": "SwiftPost", "services": ["standard", "express"], "cutoff": "15:00",
                                       "max_parcel_g": 30000, "hazmat": False},
                         "northline": {"name": "Northline Freight", "services": ["economy", "standard"],
                                       "cutoff": "17:30", "max_parcel_g": 70000, "hazmat": True},
                         "parceldirect": {"name": "ParcelDirect", "services": ["standard", "express", "next_day"],
                                          "cutoff": "13:30", "max_parcel_g": 25000, "hazmat": False},
                         "courierblue": {"name": "Courier Blue", "services": ["express", "next_day"],
                                         "cutoff": "16:00", "max_parcel_g": 20000, "hazmat": True}}},
          "skus": skus, "bins": bins, "lots": lots, "orders": orders,
          "shipments": {"SH-880": {"order": "OR-5090", "carrier": "northline", "service": "standard",
                                   "ship_date": "2026-11-06",
                                   "parcels": [{"weight_g": 33000, "length_cm": 60, "width_cm": 40, "height_cm": 40}]}},
          "adjustments": {"ADJ-40": {"lot": "LT-1102", "from_qty": 310, "to_qty": 300, "reason": "damaged",
                                     "date": "2026-11-03"}},
          "next_lot": 9000, "next_order": 5200, "next_shipment": 900, "next_adj": 50}
    for k, o in orders.items():
        for x in o["alloc"]:
            assert lots[x["lot"]]["sku"] == x["sku"], k
    for b in bins:   # one product per bin, capacity respected
        assert len(set(l["sku"] for l in lots.values() if l["bin"] == b)) <= 1, b
        assert sum(l["qty"] for l in lots.values() if l["bin"] == b) <= bins[b]["capacity"], b
    return st


def _wh_m08():
    """Variant: three yogurt orders hold stock from lot LT-1303, which is about to be recalled."""
    st = _wh()
    for k, prio, ship_by, carrier, lines, alloc in [
        ("OR-5156", "standard", "2026-11-13", "northline", [("SK-130", 10), ("SK-190", 6)],
         [("SK-130", "LT-1303", 10), ("SK-190", "LT-1901", 6)]),
        ("OR-5157", "standard", "2026-11-12", "northline", [("SK-130", 30)],
         [("SK-130", "LT-1302", 18), ("SK-130", "LT-1303", 12)]),
        ("OR-5158", "standard", "2026-11-16", "northline", [("SK-130", 40)],
         [("SK-130", "LT-1303", 18), ("SK-130", "LT-1304", 22)]),
    ]:
        st["orders"][k] = {"customer": "Cedar Market", "priority": prio, "ship_by": ship_by, "carrier": carrier,
                           "service": "standard", "status": "allocated",
                           "lines": [{"sku": s, "qty": q} for s, q in lines],
                           "alloc": [{"sku": s, "lot": l, "qty": q} for s, l, q in alloc],
                           "hold_reason": "", "hold_until": "", "held_from": "", "backorder_of": ""}
    return st


def _ship(order, carrier, service, date, parcels):
    return {"order": order, "carrier": carrier, "service": service, "ship_date": date,
            "parcels": [{"weight_g": w, "length_cm": l, "width_cm": wd, "height_cm": h} for w, l, wd, h in parcels]}


def _alloc(*xs):
    return [{"sku": s, "lot": l, "qty": q} for s, l, q in sorted(xs)]


_WT = "2026-11-10"   # warehouse "today"

# ---- single-call S07-S12
_add("S07", "warehouse", _wh(),
     "Order OR-5104 for Harbour Cafe is packed and goes out today, 10 November 2026, with Northline Freight on "
     "the standard service it is booked on. Book the shipment: two cartons - the first 18.5 kg, 60 x 40 x 40 cm; "
     "the second 7 kg 250 g, 40 x 30 x 20 cm. (Dock door 4, packer badge 2231, 14 units.)",
     [["create_shipment", {"order_id": "OR-5104", "carrier": "northline", "service": "standard",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 18500, "length_cm": 60, "width_cm": 40, "height_cm": 40},
                               {"weight_g": 7250, "length_cm": 40, "width_cm": 30, "height_cm": 20}]}]],
     [["create_shipment", {"order_id": "OR-5104", "carrier": "northline", "service": "standard",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 18500, "length_cm": 60, "width_cm": 40, "height_cm": 40},
                               {"weight_g": 7025, "length_cm": 40, "width_cm": 30, "height_cm": 20}]}]],
     lambda s, i: _final(s, i, changes={"orders": {"OR-5104": {"status": "shipped"}},
                                        "lots": {"LT-1101": {"qty": 58}, "LT-1201": {"qty": 36}}},
                         new={"shipments": [_ship("OR-5104", "northline", "standard", _WT,
                                                  [(7250, 40, 30, 20), (18500, 60, 40, 40)])]}),
     "UOXD", mentions=["Harbour Cafe", "Northline Freight"])

_add("S08", "warehouse", _wh(),
     "Cycle count in pick face BN-A03, which holds paper towels lot LT-1401. The system says 46 packs; the counter "
     "found three full dozens plus 8 loose packs on the shelf, and nothing was damaged or found elsewhere. Correct "
     "the lot to what was counted.",
     [["adjust_stock", {"lot_id": "LT-1401", "counted_qty": 44, "reason": "cycle_count"}]],
     [["adjust_stock", {"lot_id": "LT-1401", "counted_qty": 44, "reason": "damaged"}]],
     lambda s, i: _final(s, i, changes={"lots": {"LT-1401": {"qty": 44}}}, new={"adjustments": [
         {"lot": "LT-1401", "from_qty": 46, "to_qty": 44, "reason": "cycle_count", "date": _WT}]}),
     "UNT")

_add("S09", "warehouse", _wh(),
     "Allocate order OR-5110 (Green Grocer Co, ship by 11 November) as follows: the 12 bags of coffee beans "
     "SK-120 all from lot LT-1201, and the 30 yogurt trays SK-130 as 18 from lot LT-1302 plus 12 from lot LT-1303. "
     "Lot LT-1301 is too short-dated and LT-1304 is for next week, so leave those out.",
     [["allocate_stock", {"order_id": "OR-5110", "picks": [{"lot_id": "LT-1201", "qty": 12},
                                                          {"lot_id": "LT-1302", "qty": 18},
                                                          {"lot_id": "LT-1303", "qty": 12}]}]],
     [["allocate_stock", {"order_id": "OR-5110", "picks": [{"lot_id": "LT-1201", "qty": 12},
                                                          {"lot_id": "LT-1302", "qty": 18}]}]],
     lambda s, i: _final(s, i, changes={"orders": {"OR-5110": {
         "status": "allocated", "alloc": _alloc(("SK-120", "LT-1201", 12), ("SK-130", "LT-1302", 18),
                                               ("SK-130", "LT-1303", 12))}}}),
     "OX", mentions=["Green Grocer Co"])

_add("S10", "warehouse", _wh(),
     "Put order OR-5121 (Pine and Co, 10 bottles of dish soap) on hold: the delivery postcode doesn't match the "
     "street, so the address has to be verified with the customer before anything is picked. We'll look at it "
     "again on Friday 13 November 2026.",
     [["set_order_hold", {"order_id": "OR-5121", "on_hold": True, "reason": "address_check",
                          "review_on": "2026-11-13"}]],
     [["set_order_hold", {"order_id": "OR-5121", "on_hold": True, "reason": "customer_request",
                          "review_on": "2026-11-13"}]],
     lambda s, i: _final(s, i, changes={"orders": {"OR-5121": {
         "status": "on_hold", "held_from": "new", "hold_reason": "address_check", "hold_until": "2026-11-13"}}}),
     "NPD", mentions=["Pine and Co"])

_add("S11", "warehouse", _wh(),
     "Replenish the paper-towel pick face: take 40 of the 400 packs in bulk lot LT-1402 (bin BN-K03, aisle K, "
     "level 2, pallet tag 77812, received 14 October) and put them in pick face BN-A03, which already holds that "
     "product. The rest of the pallet stays in bulk.",
     [["move_stock", {"lot_id": "LT-1402", "to_bin": "BN-A03", "qty": 40}]],
     [["move_stock", {"lot_id": "LT-1402", "to_bin": "BN-A09", "qty": 40}]],
     lambda s, i: _final(s, i, changes={"lots": {"LT-1402": {"qty": 360}}}, new={"lots": [
         {"sku": "SK-140", "bin": "BN-A03", "qty": 40, "expiry": "2028-06-30", "status": "available", "q_reason": ""}]}),
     "TX")

_add("S12", "warehouse", _wh(),
     "Order OR-5130 for Bay Deli can't go out complete. Split the part we can't supply into a backorder: 8 of "
     "the 20 bottles of olive oil (SK-150) and 2 of the 6 hand sanitisers (SK-160). The 5 dish soaps (SK-190) "
     "and everything else stay on the original order.",
     [["split_order", {"order_id": "OR-5130", "lines": [{"sku": "SK-150", "qty": 8}, {"sku": "SK-160", "qty": 2}]}]],
     [["split_order", {"order_id": "OR-5130", "lines": [{"sku": "SK-150", "qty": 12}, {"sku": "SK-160", "qty": 4}]}]],
     lambda s, i: _final(s, i, changes={"orders": {"OR-5130": {"lines": [
         {"sku": "SK-150", "qty": 12}, {"sku": "SK-160", "qty": 4}, {"sku": "SK-190", "qty": 5}]}}}, new={"orders": [
             {"customer": "Bay Deli", "priority": "standard", "ship_by": "2026-11-16", "carrier": "northline",
              "service": "economy", "status": "new", "lines": [{"sku": "SK-150", "qty": 8}, {"sku": "SK-160", "qty": 2}],
              "alloc": [], "hold_reason": "", "hold_until": "", "held_from": "", "backorder_of": "OR-5130"}]}),
     "OTX", mentions=["Bay Deli"])

# ---- multi-step M07-M12
_add("M07", "warehouse", _wh(),
     "Please get order OR-5150 for Alder Foods out of the door. Allocate it following the stock rules, then book "
     "its shipment with the carrier and service already on the order: one box, 50 x 40 x 30 cm, weighing the goods "
     "plus 500 g of packaging. It should leave today if the carrier's cut-off still allows it, otherwise on the "
     "next working day. Nothing else should change.",
     [["get_warehouse_policy", {}], ["get_order", {"order_id": "OR-5150"}],
      ["find_stock", {"sku": "SK-120"}], ["find_stock", {"sku": "SK-150"}],
      ["allocate_stock", {"order_id": "OR-5150", "picks": [{"lot_id": "LT-1201", "qty": 6},
                                                          {"lot_id": "LT-1501", "qty": 4}]}],
      ["get_order", {"order_id": "OR-5150"}],
      ["create_shipment", {"order_id": "OR-5150", "carrier": "parceldirect", "service": "standard",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 12000, "length_cm": 50, "width_cm": 40, "height_cm": 30}]}],
      ["create_shipment", {"order_id": "OR-5150", "carrier": "parceldirect", "service": "standard",
                           "ship_date": "2026-11-11", "parcels": [
                               {"weight_g": 12000, "length_cm": 50, "width_cm": 40, "height_cm": 30}]}]],
     [["allocate_stock", {"order_id": "OR-5150", "picks": [{"lot_id": "LT-1202", "qty": 6},
                                                          {"lot_id": "LT-1502", "qty": 4}]}],
      ["allocate_stock", {"order_id": "OR-5150", "picks": [{"lot_id": "LT-1203", "qty": 6},
                                                          {"lot_id": "LT-1503", "qty": 4}]}],
      ["allocate_stock", {"order_id": "OR-5150", "picks": [{"lot_id": "LT-1201", "qty": 6},
                                                          {"lot_id": "LT-1501", "qty": 4}]}]],
     lambda s, i: _final(s, i, changes={
         "orders": {"OR-5150": {"status": "shipped", "alloc": _alloc(("SK-120", "LT-1201", 6), ("SK-150", "LT-1501", 4))}},
         "lots": {"LT-1201": {"qty": 34}, "LT-1501": {"qty": 16}}},
         new={"shipments": [_ship("OR-5150", "parceldirect", "standard", "2026-11-11", [(12000, 50, 40, 30)])]}),
     "abcef", ref_errors=[6], mentions=["Alder Foods"])

_add("M08", "warehouse", _wh_m08(),
     "Our yogurt supplier has recalled lot LT-1303. Quarantine it for the recall. Every order holding stock from "
     "that lot must have its whole allocation redone from other stock, following the stock rules; redo them in "
     "order of ship-by date (earliest first). An order that can no longer be covered completely is left with "
     "nothing allocated and put on hold for a stock issue instead. Change nothing else.",
     [["get_warehouse_policy", {}], ["find_stock", {"sku": "SK-130"}], ["list_orders", {"status": "allocated"}],
      ["quarantine_lot", {"lot_id": "LT-1303", "reason": "recall"}],
      ["release_allocation", {"order_id": "OR-5156"}], ["release_allocation", {"order_id": "OR-5157"}],
      ["release_allocation", {"order_id": "OR-5158"}],
      ["quarantine_lot", {"lot_id": "LT-1303", "reason": "recall"}],
      ["find_stock", {"sku": "SK-130"}], ["get_order", {"order_id": "OR-5156"}],
      ["allocate_stock", {"order_id": "OR-5157", "picks": [{"lot_id": "LT-1302", "qty": 18},
                                                          {"lot_id": "LT-1305", "qty": 12}]}],
      ["allocate_stock", {"order_id": "OR-5156", "picks": [{"lot_id": "LT-1305", "qty": 10},
                                                          {"lot_id": "LT-1901", "qty": 6}]}],
      ["set_order_hold", {"order_id": "OR-5158", "on_hold": True, "reason": "stock_issue"}]],
     [["release_allocation", {"order_id": "OR-5156"}], ["release_allocation", {"order_id": "OR-5157"}],
      ["release_allocation", {"order_id": "OR-5158"}], ["quarantine_lot", {"lot_id": "LT-1303", "reason": "recall"}],
      ["allocate_stock", {"order_id": "OR-5156", "picks": [{"lot_id": "LT-1302", "qty": 10},
                                                          {"lot_id": "LT-1901", "qty": 6}]}],
      ["allocate_stock", {"order_id": "OR-5157", "picks": [{"lot_id": "LT-1302", "qty": 8},
                                                          {"lot_id": "LT-1305", "qty": 22}]}],
      ["allocate_stock", {"order_id": "OR-5158", "picks": [{"lot_id": "LT-1305", "qty": 3},
                                                          {"lot_id": "LT-1304", "qty": 30}]}]],
     lambda s, i: _final(s, i, changes={
         "lots": {"LT-1303": {"status": "quarantine", "q_reason": "recall"}},
         "orders": {"OR-5157": {"alloc": _alloc(("SK-130", "LT-1302", 18), ("SK-130", "LT-1305", 12))},
                    "OR-5156": {"alloc": _alloc(("SK-130", "LT-1305", 10), ("SK-190", "LT-1901", 6))},
                    "OR-5158": {"status": "on_hold", "alloc": [], "hold_reason": "stock_issue", "held_from": "new"}}}),
     "abcdf", ref_errors=[3], seq=[[10, 11]])

_add("M09", "warehouse", _wh(),
     "It's late in the day. Ship every allocated order whose ship-by date is today, each as one parcel of "
     "40 x 30 x 30 cm weighing the goods plus 400 g. If the order's carrier has already passed today's cut-off, "
     "first switch it to the carrier with the latest cut-off today that offers the same service and can carry "
     "it, then ship it today with that one. Orders on hold, orders not fully allocated and orders due later "
     "stay as they are.",
     [["get_warehouse_policy", {}], ["list_orders", {"status": "allocated", "ship_by": "2026-11-10"}],
      ["get_order", {"order_id": "OR-5161"}], ["get_order", {"order_id": "OR-5162"}],
      ["get_order", {"order_id": "OR-5163"}],
      ["create_shipment", {"order_id": "OR-5161", "carrier": "swiftpost", "service": "standard",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 5750, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
      ["create_shipment", {"order_id": "OR-5162", "carrier": "parceldirect", "service": "express",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 6900, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
      ["change_carrier", {"order_id": "OR-5162", "carrier": "courierblue", "service": "express"}],
      ["create_shipment", {"order_id": "OR-5162", "carrier": "courierblue", "service": "express",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 6900, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
      ["change_carrier", {"order_id": "OR-5163", "carrier": "northline", "service": "standard"}],
      ["create_shipment", {"order_id": "OR-5163", "carrier": "northline", "service": "standard",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 8800, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
      ["list_orders", {"status": "allocated", "ship_by": "2026-11-10"}]],
     [["create_shipment", {"order_id": "OR-5161", "carrier": "swiftpost", "service": "standard",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 5750, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
      ["change_carrier", {"order_id": "OR-5162", "carrier": "swiftpost", "service": "express"}],
      ["create_shipment", {"order_id": "OR-5162", "carrier": "swiftpost", "service": "express",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 6900, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
      ["change_carrier", {"order_id": "OR-5163", "carrier": "swiftpost", "service": "standard"}]],
     lambda s, i: _final(s, i, changes={
         "orders": {"OR-5161": {"status": "shipped"},
                    "OR-5162": {"status": "shipped", "carrier": "courierblue"},
                    "OR-5163": {"status": "shipped", "carrier": "northline"}},
         "lots": {"LT-1201": {"qty": 37}, "LT-1901": {"qty": 78}, "LT-1501": {"qty": 15}, "LT-1401": {"qty": 42}}},
         new={"shipments": [_ship("OR-5161", "swiftpost", "standard", _WT, [(5750, 40, 30, 30)]),
                            _ship("OR-5162", "courierblue", "express", _WT, [(6900, 40, 30, 30)]),
                            _ship("OR-5163", "northline", "standard", _WT, [(8800, 40, 30, 30)])]}),
     "abcdef", ref_errors=[6], seq=[[7, 8], [9, 10]])

_add("M10", "warehouse", _wh(),
     "Cycle-count results from aisle A: lot LT-1101 counted 52, LT-1201 counted 40, LT-1501 counted 3 and LT-1901 "
     "counted 80. Correct every lot whose count differs from the system, as a cycle count. If an order has more "
     "allocated from a lot than was counted, release that order's allocation first and, once the count is "
     "corrected, allocate the order again following the stock rules. Nothing else should change.",
     [["get_warehouse_policy", {}], ["list_bins", {"aisle": "A"}],
      ["adjust_stock", {"lot_id": "LT-1101", "counted_qty": 52, "reason": "cycle_count"}],
      ["adjust_stock", {"lot_id": "LT-1501", "counted_qty": 3, "reason": "cycle_count"}],
      ["find_stock", {"sku": "SK-150"}], ["release_allocation", {"order_id": "OR-5162"}],
      ["adjust_stock", {"lot_id": "LT-1501", "counted_qty": 3, "reason": "cycle_count"}],
      ["find_stock", {"sku": "SK-150"}], ["get_order", {"order_id": "OR-5162"}],
      ["allocate_stock", {"order_id": "OR-5162", "picks": [{"lot_id": "LT-1501", "qty": 3},
                                                          {"lot_id": "LT-1503", "qty": 2}]}]],
     [["adjust_stock", {"lot_id": "LT-1101", "counted_qty": 52, "reason": "cycle_count"}],
      ["release_allocation", {"order_id": "OR-5162"}],
      ["adjust_stock", {"lot_id": "LT-1501", "counted_qty": 3, "reason": "cycle_count"}],
      ["allocate_stock", {"order_id": "OR-5162", "picks": [{"lot_id": "LT-1501", "qty": 3},
                                                          {"lot_id": "LT-1502", "qty": 2}]}],
      ["allocate_stock", {"order_id": "OR-5162", "picks": [{"lot_id": "LT-1501", "qty": 3}]}]],
     lambda s, i: _final(s, i, changes={
         "lots": {"LT-1101": {"qty": 52}, "LT-1501": {"qty": 3}},
         "orders": {"OR-5162": {"alloc": _alloc(("SK-150", "LT-1501", 3), ("SK-150", "LT-1503", 2))}}},
         new={"adjustments": [
             {"lot": "LT-1101", "from_qty": 60, "to_qty": 52, "reason": "cycle_count", "date": _WT},
             {"lot": "LT-1501", "from_qty": 20, "to_qty": 3, "reason": "cycle_count", "date": _WT}]}),
     "abcdg", ref_errors=[3], seq=[[5, 6, 9]])

_add("M11", "warehouse", _wh(),
     "Copper Kettle's order OR-5170 can't be filled completely from stock we may use. Move whatever can't be "
     "supplied now into a backorder, allocate the rest following the stock rules, and ship it today with the "
     "carrier and service on the order, as one box of 60 x 40 x 40 cm weighing the goods plus 800 g of "
     "packaging. The backorder stays unallocated. Nothing else should change.",
     [["get_warehouse_policy", {}], ["get_order", {"order_id": "OR-5170"}],
      ["find_stock", {"sku": "SK-180"}], ["find_stock", {"sku": "SK-190"}],
      ["split_order", {"order_id": "OR-5170", "lines": [{"sku": "SK-180", "qty": 10}]}],
      ["get_order", {"order_id": "OR-5170"}],
      ["allocate_stock", {"order_id": "OR-5170", "picks": [{"lot_id": "LT-1801", "qty": 25},
                                                          {"lot_id": "LT-1802", "qty": 35},
                                                          {"lot_id": "LT-1901", "qty": 4}]}],
      ["create_shipment", {"order_id": "OR-5170", "carrier": "northline", "service": "standard",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 30400, "length_cm": 60, "width_cm": 40, "height_cm": 40}]}]],
     [["allocate_stock", {"order_id": "OR-5170", "picks": [{"lot_id": "LT-1801", "qty": 25},
                                                          {"lot_id": "LT-1802", "qty": 35},
                                                          {"lot_id": "LT-1901", "qty": 4}]}],
      ["split_order", {"order_id": "OR-5170", "lines": [{"sku": "SK-180", "qty": 10}]}],
      ["split_order", {"order_id": "OR-5170", "lines": [{"sku": "SK-180", "qty": 22}]}]],
     lambda s, i: _final(s, i, changes={
         "orders": {"OR-5170": {"status": "shipped", "lines": [{"sku": "SK-180", "qty": 60}, {"sku": "SK-190", "qty": 4}],
                                "alloc": _alloc(("SK-180", "LT-1801", 25), ("SK-180", "LT-1802", 35),
                                                ("SK-190", "LT-1901", 4))}},
         "lots": {"LT-1801": {"qty": 0}, "LT-1802": {"qty": 0}, "LT-1901": {"qty": 76}}},
         new={"orders": [{"customer": "Copper Kettle", "priority": "standard", "ship_by": "2026-11-13",
                          "carrier": "northline", "service": "standard", "status": "new",
                          "lines": [{"sku": "SK-180", "qty": 10}], "alloc": [], "hold_reason": "", "hold_until": "",
                          "held_from": "", "backorder_of": "OR-5170"}],
              "shipments": [_ship("OR-5170", "northline", "standard", _WT, [(30400, 60, 40, 40)])]}),
     "acef", mentions=["Copper Kettle"], seq=[[4, 6, 7]])

_add("M12", "warehouse", _wh(),
     "Studio Nine's order OR-5175 needs to go out today as one box of 40 x 30 x 30 cm weighing the goods plus "
     "300 g. If the carrier it is booked with can't take it, switch it to one that can, keeping the same service "
     "and still making today's cut-off - if several can, the one with the latest cut-off - and ship with that. "
     "Nothing else should change.",
     [["get_warehouse_policy", {}], ["get_order", {"order_id": "OR-5175"}],
      ["create_shipment", {"order_id": "OR-5175", "carrier": "swiftpost", "service": "express",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 7200, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
      ["change_carrier", {"order_id": "OR-5175", "carrier": "courierblue", "service": "express"}],
      ["get_order", {"order_id": "OR-5175"}],
      ["create_shipment", {"order_id": "OR-5175", "carrier": "courierblue", "service": "express",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 7200, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}]],
     [["change_carrier", {"order_id": "OR-5175", "carrier": "northline", "service": "express"}],
      ["change_carrier", {"order_id": "OR-5175", "carrier": "courierblue", "service": "next_day"}],
      ["create_shipment", {"order_id": "OR-5175", "carrier": "courierblue", "service": "next_day",
                           "ship_date": "2026-11-10", "parcels": [
                               {"weight_g": 7200, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}]],
     lambda s, i: _final(s, i, changes={
         "orders": {"OR-5175": {"status": "shipped", "carrier": "courierblue"}},
         "lots": {"LT-1401": {"qty": 44}, "LT-1701": {"qty": 24}}},
         new={"shipments": [_ship("OR-5175", "courierblue", "express", _WT, [(7200, 40, 30, 30)])]}),
     "abcef", ref_errors=[2], mentions=["Studio Nine"], seq=[[3, 5]])

# ---- calibration C2 (multi), C7 (single)
_add("C2", "warehouse", _wh(),
     "Fridge bay BN-C01 is failing and must be emptied today. Taking its lots in order of lot id, move each one, whole, "
     "into another chilled bin: one that already holds the same product and has room, the lowest-numbered such bin first; failing that, the "
     "lowest-numbered empty chilled bin. A lot from BN-C01 that expires before today plus the minimum shelf life "
     "for chilled goods is quarantined for the temperature problem instead of moved. Nothing else should change.",
     [["get_warehouse_policy", {}], ["list_bins", {"zone": "chilled"}], ["find_stock", {"sku": "SK-130"}],
      ["quarantine_lot", {"lot_id": "LT-1301", "reason": "temperature_breach"}],
      ["move_stock", {"lot_id": "LT-1302", "to_bin": "BN-C02", "qty": 18}],
      ["move_stock", {"lot_id": "LT-1305", "to_bin": "BN-C02", "qty": 25}],
      ["move_stock", {"lot_id": "LT-1305", "to_bin": "BN-C03", "qty": 25}],
      ["list_bins", {"zone": "chilled"}]],
     [["move_stock", {"lot_id": "LT-1301", "to_bin": "BN-C02", "qty": 10}],
      ["move_stock", {"lot_id": "LT-1302", "to_bin": "BN-C02", "qty": 18}],
      ["move_stock", {"lot_id": "LT-1305", "to_bin": "BN-C06", "qty": 25}]],
     lambda s, i: _final(s, i, changes={"lots": {
         "LT-1301": {"status": "quarantine", "q_reason": "temperature_breach"},
         "LT-1302": {"bin": "BN-C02"}, "LT-1305": {"bin": "BN-C03"}}}),
     "abdf", ref_errors=[5])

_add("C7", "warehouse", _wh(),
     "The temperature logger in chilled bin BN-C04 recorded 11 degrees for three hours overnight (alarm 23:40 to "
     "02:45). The fresh pasta in there, lot LT-1801 (25 packs, best before 18 November), must not be picked until "
     "quality has looked at it. Block that lot. Lot LT-1802 next door in BN-C05 was fine.",
     [["quarantine_lot", {"lot_id": "LT-1801", "reason": "temperature_breach"}]],
     [["quarantine_lot", {"lot_id": "LT-1802", "reason": "temperature_breach"}]],
     lambda s, i: _final(s, i, changes={"lots": {"LT-1801": {"status": "quarantine", "q_reason": "temperature_breach"}}}),
     "NTX")


# --------------------------------------------------------------------------
# 3. research-lab sample custody: samples, freezers/racks, chain of custody, retention and destruction
# --------------------------------------------------------------------------
_SAMPLE_TYPES = ["plasma", "serum", "rna", "urine", "dna", "tissue", "cells"]
_SAMPLE_STATUS = ["stored", "checked_out", "destroyed"]
_PURPOSES = ["assay", "aliquoting", "shipment_prep", "quality_check"]
_TRANSFER_REASONS = ["handover", "courier", "shift_change"]
_DESTROY_METHODS = ["autoclave", "incineration", "chemical"]
_HOLD_KINDS = ["legal", "quality", "audit"]


class LabEnv(_Env):
    TOOLS = [
        _fn("get_lab_policy", "Today's date, storage temperature per sample type, permitted destruction methods, "
            "and the custody, storage, retention and destruction rules that the write tools enforce."),
        _fn("list_studies", "All studies: status, closing date, retention period, date retention ends, sample "
            "custodian and principal investigator."),
        _fn("list_staff", "All staff: role, training, status and supervisor."),
        _fn("find_samples", "Samples, filtered, by sample id; 10 per page.",
            {"study_id": _s("Only this study, e.g. ST-01."), "participant_id": _s("Only this participant, e.g. PT-101."),
             "type": _s("Only this sample type.", _SAMPLE_TYPES), "status": _s("Only this status.", _SAMPLE_STATUS),
             "freezer_id": _s("Only samples stored in this freezer, e.g. FZ-04."),
             "custodian_id": _s("Only samples in this person's custody, e.g. PS-07."), "page": _PAGE}),
        _fn("get_sample", "One sample in detail, with its study, participant consent, location and custody.",
            {"sample_id": _s("Sample id, e.g. SM-0412.")}, ["sample_id"]),
        _fn("list_storage", "Without freezer_id: every freezer with temperature and status. With freezer_id: its "
            "racks, which slot holds which sample, and the free slots.",
            {"freezer_id": _s("Freezer id, e.g. FZ-05.")}),
        _fn("relocate_sample", "Move a stored sample to another rack slot (within storage).",
            {"sample_id": _s("Sample id."), "rack_id": _s("Destination rack, e.g. RK-41."),
             "slot": _i("Destination slot number.", 1, 100)}, ["sample_id", "rack_id", "slot"]),
        _fn("check_out_sample", "Take a stored sample out of storage into a person's custody.",
            {"sample_id": _s("Sample id."), "to_person": _s("Staff id who takes custody."),
             "purpose": _s("What it is taken out for.", _PURPOSES), "due_back": _s("Return date, YYYY-MM-DD.")},
            ["sample_id", "to_person", "purpose", "due_back"]),
        _fn("return_sample", "Put a checked-out sample back into storage; custody returns to the study's custodian.",
            {"sample_id": _s("Sample id."), "rack_id": _s("Rack id."), "slot": _i("Slot number.", 1, 100),
             "volume_used_ul": _i("Optional microlitres used while it was out (default 0).", 0, 100000)},
            ["sample_id", "rack_id", "slot"]),
        _fn("transfer_custody", "Hand checked-out samples from one person to another (chain of custody).",
            {"sample_ids": _a("Samples handed over.", _s("Sample id."), 1, 20),
             "from_person": _s("Staff id handing over."), "to_person": _s("Staff id receiving."),
             "reason": _s("Why custody changes.", _TRANSFER_REASONS)},
            ["sample_ids", "from_person", "to_person", "reason"]),
        _fn("create_aliquots", "Split part of a stored sample into new child samples (aliquots) in storage.",
            {"sample_id": _s("Parent sample id."),
             "aliquots": _a("The aliquots to make.", _o({"volume_ul": _i("Volume in microlitres.", 1, 100000),
                                                         "rack_id": _s("Rack id."),
                                                         "slot": _i("Slot number.", 1, 100)},
                                                        ["volume_ul", "rack_id", "slot"]), 1, 8)},
            ["sample_id", "aliquots"]),
        _fn("destroy_samples", "Destroy stored samples, with a witness.",
            {"sample_ids": _a("Samples to destroy.", _s("Sample id."), 1, 20),
             "method": _s("Destruction method.", _DESTROY_METHODS), "witness_id": _s("Staff id of the witness.")},
            ["sample_ids", "method", "witness_id"]),
        _fn("set_sample_hold", "Place a hold on a sample (on_hold=true, with a reason) or lift it (on_hold=false).",
            {"sample_id": _s("Sample id."), "on_hold": _b("true to place, false to lift."),
             "reason": _s("Kind of hold (required when placing).", _HOLD_KINDS),
             "review_on": _s("Optional date to review the hold, YYYY-MM-DD.")}, ["sample_id", "on_hold"]),
        _fn("record_consent_withdrawal", "Record that a participant has withdrawn consent (applies to all of "
            "their samples).",
            {"participant_id": _s("Participant id."), "date_received": _s("Date the withdrawal was received.")},
            ["participant_id", "date_received"]),
    ]

    def _sample(self, sid):
        return self._get("samples", sid, "sample")

    def _person(self, pid):
        p = self._get("staff", pid, "staff member")
        if p["status"] != "active":
            raise _Rule("%s (%s) is %s" % (pid, p["name"], p["status"]))
        return p

    def _temp(self, s):
        return self.state["policy"]["storage_temp"][s["type"]]

    def _retention_end(self, st):
        if st["status"] != "closed":
            return ""
        y, rest = int(st["closed_on"][:4]), st["closed_on"][4:]
        if rest == "-02-29":
            rest = "-02-28"
        return "%04d%s" % (y + st["retention_years"], rest)

    def _occupant(self, rack, slot):
        for k, s in sorted(self.state["samples"].items()):
            if s["status"] == "stored" and s["rack"] == rack and s["slot"] == slot:
                return k
        return ""

    def _place(self, s, rack, slot, taken=()):
        r = self._get("racks", rack, "rack")
        fz = self.state["freezers"][r["freezer"]]
        if fz["temp"] != self._temp(s):
            raise _Rule("%s samples are stored at %d C; rack %s is in %s at %d C" % (
                s["type"], self._temp(s), rack, r["freezer"], fz["temp"]))
        if fz["status"] != "ok":
            raise _Rule("freezer %s is in %s and cannot receive samples" % (r["freezer"], fz["status"]))
        if slot > r["slots"]:
            raise _Rule("rack %s has slots 1-%d" % (rack, r["slots"]))
        occ = self._occupant(rack, slot)
        if occ or (rack, slot) in taken:
            raise _Rule("slot %d of rack %s is taken%s" % (slot, rack, " by " + occ if occ else ""))

    def _log(self, samples, frm, to, reason):
        k = self._new_id("next_custody", "CT")
        self.state["custody_log"][k] = {"samples": sorted(samples), "from": frm, "to": to,
                                        "date": self.state["today"], "reason": reason}
        return k

    def _row(self, k):
        s = self.state["samples"][k]
        row = dict(sample_id=k, **s)
        row["freezer"] = self.state["racks"][s["rack"]]["freezer"] if s["rack"] else ""
        row["consent"] = self.state["participants"][s["participant"]]["consent"]
        row["storage_temp"] = self._temp(s)
        return row

    def _t_get_lab_policy(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "storage_temp_c": p["storage_temp"],
                "destruction_methods": p["methods"], "max_checkout_days": p["max_checkout_days"],
                "min_parent_ul": p["min_parent_ul"], "min_aliquot_ul": p["min_aliquot_ul"],
                "rules": [
                    "Samples may only be stored in a freezer at the storage temperature for their type, and only "
                    "freezers with status ok can receive samples (moving samples out of any freezer is allowed). "
                    "One sample per slot.",
                    "Stored samples are in the custody of their study's custodian. Checking a sample out puts it "
                    "in the custody of the person taking it; due_back must be after today and at most "
                    "max_checkout_days after today. Samples on hold, samples whose participant has withdrawn "
                    "consent, and destroyed samples cannot be checked out or aliquoted.",
                    "Anyone taking custody (check-out or transfer) must be active and trained in biohazard, and in "
                    "ln2 for samples stored at -196 C. A transfer with reason courier instead needs a person "
                    "trained as a courier. Only the current custodian can hand samples over, and only checked-out "
                    "samples can be transferred.",
                    "Aliquots: each at least min_aliquot_ul; the parent must keep at least min_parent_ul.",
                    "A stored sample may be destroyed only if its participant has withdrawn consent or its study's "
                    "retention period has ended (closing date plus retention_years), and never while on hold. "
                    "The method must be one permitted for the sample type. The witness must be active, trained in "
                    "destruction, and not the custodian of any of the samples destroyed in that call.",
                ]}

    def _t_list_studies(self, a):
        return {"studies": [dict(study_id=k, retention_ends=self._retention_end(st), **st)
                            for k, st in sorted(self.state["studies"].items())]}

    def _t_list_staff(self, a):
        return {"staff": [dict(staff_id=k, **p) for k, p in sorted(self.state["staff"].items())]}

    def _t_find_samples(self, a):
        for key, coll, what in (("study_id", "studies", "study"), ("participant_id", "participants", "participant"),
                                ("freezer_id", "freezers", "freezer"), ("custodian_id", "staff", "staff member")):
            if key in a:
                self._get(coll, a[key], what)
        rows = []
        for k, s in sorted(self.state["samples"].items()):
            fz = self.state["racks"][s["rack"]]["freezer"] if s["rack"] else ""
            if a.get("study_id", s["study"]) == s["study"] and a.get("participant_id", s["participant"]) == s["participant"] \
                    and a.get("type", s["type"]) == s["type"] and a.get("status", s["status"]) == s["status"] \
                    and a.get("freezer_id", fz) == fz and a.get("custodian_id", s["custodian"]) == s["custodian"] \
                    and ("freezer_id" not in a or s["status"] == "stored"):
                rows.append(k)
        meta, rows = _paged(rows, a.get("page", 1), 10)
        meta["samples"] = [self._row(k) for k in rows]
        return meta

    def _t_get_sample(self, a):
        k = a["sample_id"]
        s = self._sample(k)
        row = self._row(k)
        st = self.state["studies"][s["study"]]
        row["study"] = dict(study_id=s["study"], retention_ends=self._retention_end(st), **st)
        row["participant"] = dict(participant_id=s["participant"], **self.state["participants"][s["participant"]])
        row["children"] = sorted(c for c, x in self.state["samples"].items() if x["parent"] == k)
        return row

    def _t_list_storage(self, a):
        if "freezer_id" not in a:
            return {"freezers": [dict(freezer_id=k, racks=sorted(r for r, x in self.state["racks"].items()
                                                                 if x["freezer"] == k), **f)
                                 for k, f in sorted(self.state["freezers"].items())]}
        fid = a["freezer_id"]
        f = self._get("freezers", fid, "freezer")
        racks = []
        for r, x in sorted(self.state["racks"].items()):
            if x["freezer"] != fid:
                continue
            occ = dict((s["slot"], k) for k, s in sorted(self.state["samples"].items())
                       if s["status"] == "stored" and s["rack"] == r)
            racks.append({"rack_id": r, "slots": x["slots"],
                          "occupied": [{"slot": n, "sample_id": occ[n]} for n in sorted(occ)],
                          "free_slots": [n for n in range(1, x["slots"] + 1) if n not in occ]})
        return dict(freezer_id=fid, racks_detail=racks, **f)

    def _t_relocate_sample(self, a):
        k = a["sample_id"]
        s = self._sample(k)
        if s["status"] != "stored":
            raise _Rule("sample %s is %s; only stored samples can be relocated" % (k, s["status"]))
        if (s["rack"], s["slot"]) == (a["rack_id"], a["slot"]):
            raise _Rule("sample %s is already in %s slot %d" % (k, s["rack"], s["slot"]))
        self._place(s, a["rack_id"], a["slot"])
        s["rack"], s["slot"] = a["rack_id"], a["slot"]
        return {"ok": True, "sample": self._row(k)}

    def _usable(self, k, s):
        if s["status"] != "stored":
            raise _Rule("sample %s is %s, not stored" % (k, s["status"]))
        if s["hold"]:
            raise _Rule("sample %s is on %s hold" % (k, s["hold"]))
        if self.state["participants"][s["participant"]]["consent"] != "given":
            raise _Rule("participant %s has withdrawn consent; sample %s cannot be used" % (s["participant"], k))

    def _can_hold(self, pid, s):
        p = self._person(pid)
        if "biohazard" not in p["training"]:
            raise _Rule("%s (%s) is not trained in biohazard handling" % (pid, p["name"]))
        if self._temp(s) == -196 and "ln2" not in p["training"]:
            raise _Rule("%s (%s) is not trained to handle samples stored at -196 C (ln2)" % (pid, p["name"]))

    def _t_check_out_sample(self, a):
        k = a["sample_id"]
        s = self._sample(k)
        self._usable(k, s)
        self._can_hold(a["to_person"], s)
        d = _pdate(a["due_back"], "due_back")
        last = _plus_days(self.state["today"], self.state["policy"]["max_checkout_days"])
        if not (self.state["today"] < a["due_back"] <= last):
            raise _Rule("due_back must be after today and no later than %s" % last)
        self._log([k], s["custodian"], a["to_person"], "check_out")
        s.update({"status": "checked_out", "rack": "", "slot": 0, "custodian": a["to_person"],
                  "purpose": a["purpose"], "due_back": d.isoformat()})
        return {"ok": True, "sample": self._row(k)}

    def _t_return_sample(self, a):
        k = a["sample_id"]
        s = self._sample(k)
        if s["status"] != "checked_out":
            raise _Rule("sample %s is %s, not checked out" % (k, s["status"]))
        used = a.get("volume_used_ul", 0)
        if used > s["volume_ul"]:
            raise _Rule("sample %s holds only %d ul" % (k, s["volume_ul"]))
        self._place(s, a["rack_id"], a["slot"])
        home = self.state["studies"][s["study"]]["custodian"]
        self._log([k], s["custodian"], home, "return")
        s.update({"status": "stored", "rack": a["rack_id"], "slot": a["slot"], "custodian": home, "purpose": "",
                  "due_back": "", "volume_ul": s["volume_ul"] - used})
        return {"ok": True, "sample": self._row(k)}

    def _t_transfer_custody(self, a):
        ids = a["sample_ids"]
        if len(set(ids)) != len(ids):
            raise _Rule("a sample is listed twice")
        frm, to = a["from_person"], a["to_person"]
        self._get("staff", frm, "staff member")
        if frm == to:
            raise _Rule("from_person and to_person are the same")
        for k in ids:
            s = self._sample(k)
            if s["status"] != "checked_out":
                raise _Rule("sample %s is %s; only checked-out samples can be handed over" % (k, s["status"]))
            if s["custodian"] != frm:
                raise _Rule("sample %s is in the custody of %s, not %s" % (k, s["custodian"], frm))
            if a["reason"] == "courier":
                p = self._person(to)
                if "courier" not in p["training"]:
                    raise _Rule("%s (%s) is not trained as a courier" % (to, p["name"]))
            else:
                self._can_hold(to, s)
        self._log(ids, frm, to, a["reason"])
        for k in ids:
            self.state["samples"][k]["custodian"] = to
        return {"ok": True, "samples": [self._row(k) for k in sorted(ids)]}

    def _t_create_aliquots(self, a):
        k = a["sample_id"]
        s = self._sample(k)
        self._usable(k, s)
        p = self.state["policy"]
        taken, total = [], 0
        for al in a["aliquots"]:
            if al["volume_ul"] < p["min_aliquot_ul"]:
                raise _Rule("aliquots must be at least %d ul" % p["min_aliquot_ul"])
            self._place(s, al["rack_id"], al["slot"], taken)
            taken.append((al["rack_id"], al["slot"]))
            total += al["volume_ul"]
        if s["volume_ul"] - total < p["min_parent_ul"]:
            raise _Rule("sample %s has %d ul; after %d ul of aliquots it would keep %d ul, below the minimum of %d ul" % (
                k, s["volume_ul"], total, s["volume_ul"] - total, p["min_parent_ul"]))
        made = []
        for al in a["aliquots"]:
            n = self._new_id("next_sample", "SM")
            self.state["samples"][n] = {
                "study": s["study"], "participant": s["participant"], "type": s["type"], "volume_ul": al["volume_ul"],
                "collected": s["collected"], "status": "stored", "rack": al["rack_id"], "slot": al["slot"],
                "custodian": self.state["studies"][s["study"]]["custodian"], "hold": "", "hold_review": "",
                "parent": k, "purpose": "", "due_back": "", "destroyed_on": "", "destroy_method": "", "witness": ""}
            made.append(n)
        s["volume_ul"] -= total
        return {"ok": True, "parent": self._row(k), "aliquots": [self._row(n) for n in made]}

    def _t_destroy_samples(self, a):
        ids = a["sample_ids"]
        if len(set(ids)) != len(ids):
            raise _Rule("a sample is listed twice")
        w = self._person(a["witness_id"])
        if "destruction" not in w["training"]:
            raise _Rule("%s (%s) is not trained to witness destruction" % (a["witness_id"], w["name"]))
        for k in ids:
            s = self._sample(k)
            if s["status"] != "stored":
                raise _Rule("sample %s is %s; only stored samples can be destroyed" % (k, s["status"]))
            if s["hold"]:
                raise _Rule("sample %s is on %s hold" % (k, s["hold"]))
            st = self.state["studies"][s["study"]]
            end = self._retention_end(st)
            if self.state["participants"][s["participant"]]["consent"] != "withdrawn" and \
                    not (end and end <= self.state["today"]):
                raise _Rule("sample %s: consent is in place and the retention of %s %s" % (
                    k, s["study"], "runs until %s" % end if end else "has not started (study still active)"))
            if a["method"] not in self.state["policy"]["methods"][s["type"]]:
                raise _Rule("%s samples must be destroyed by %s" % (s["type"], " or ".join(
                    self.state["policy"]["methods"][s["type"]])))
            if s["custodian"] == a["witness_id"]:
                raise _Rule("%s is the custodian of %s and cannot witness its destruction" % (a["witness_id"], k))
        for k in ids:
            self.state["samples"][k].update({"status": "destroyed", "rack": "", "slot": 0,
                                             "destroyed_on": self.state["today"], "destroy_method": a["method"],
                                             "witness": a["witness_id"]})
        return {"ok": True, "destroyed": sorted(ids)}

    def _t_set_sample_hold(self, a):
        k = a["sample_id"]
        s = self._sample(k)
        if s["status"] == "destroyed":
            raise _Rule("sample %s is destroyed" % k)
        if a["on_hold"]:
            if s["hold"]:
                raise _Rule("sample %s is already on %s hold" % (k, s["hold"]))
            if "reason" not in a:
                raise _Rule("a reason is required to place a hold")
            if "review_on" in a and _pdate(a["review_on"], "review_on") <= self._today():
                raise _Rule("review_on must be after today")
            s["hold"], s["hold_review"] = a["reason"], a.get("review_on", "")
        else:
            if not s["hold"]:
                raise _Rule("sample %s is not on hold" % k)
            if "reason" in a or "review_on" in a:
                raise _Rule("reason and review_on are only used when placing a hold")
            s["hold"], s["hold_review"] = "", ""
        return {"ok": True, "sample": self._row(k)}

    def _t_record_consent_withdrawal(self, a):
        pid = a["participant_id"]
        p = self._get("participants", pid, "participant")
        if p["consent"] != "given":
            raise _Rule("participant %s already withdrew consent on %s" % (pid, p["withdrawn_on"]))
        if _pdate(a["date_received"], "date_received") > self._today():
            raise _Rule("date_received is in the future")
        p["consent"], p["withdrawn_on"] = "withdrawn", a["date_received"]
        return {"ok": True, "participant_id": pid, "samples": sorted(
            k for k, s in self.state["samples"].items() if s["participant"] == pid and s["status"] != "destroyed")}


def _lab():
    staff = {}
    for k, name, role, training, status, sup in [
        ("PS-01", "Dr Helen Brooks", "lab manager", ["biohazard", "ln2", "destruction"], "active", ""),
        ("PS-02", "Omar Reyes", "technician", ["biohazard"], "active", "PS-01"),
        ("PS-03", "Kim Tran", "biobank technician", ["biohazard", "ln2", "destruction"], "active", "PS-01"),
        ("PS-04", "Dr Mira Patel", "scientist", ["biohazard"], "active", "PS-01"),
        ("PS-05", "Jonas Weber", "technician", ["biohazard", "destruction"], "active", "PS-03"),
        ("PS-06", "Leo Martins", "scientist", ["biohazard", "ln2"], "active", "PS-04"),
        ("PS-07", "Anika Rao", "technician", ["biohazard", "ln2"], "active", "PS-04"),
        ("PS-08", "Tomas Ruiz", "technician", ["biohazard"], "active", "PS-03"),
        ("PS-09", "Ravi Shah", "scientist", ["biohazard"], "active", "PS-01"),
        ("PS-10", "Pat Doyle", "courier", ["courier"], "active", ""),
        ("PS-11", "Sam Okoro", "courier", ["courier", "biohazard"], "active", ""),
        ("PS-12", "Eva Lind", "technician", ["biohazard", "ln2", "destruction"], "left", "PS-01"),
    ]:
        staff[k] = {"name": name, "role": role, "training": training, "status": status, "supervisor": sup}
    studies = {
        "ST-01": {"title": "Cardiac Biomarkers Cohort", "pi": "PS-04", "status": "active", "closed_on": "",
                  "retention_years": 10, "custodian": "PS-03"},
        "ST-02": {"title": "Asthma Panel 2006", "pi": "PS-01", "status": "closed", "closed_on": "2016-06-30",
                  "retention_years": 10, "custodian": "PS-01"},
        "ST-03": {"title": "Liver Fibrosis Biopsy Study", "pi": "PS-06", "status": "active", "closed_on": "",
                  "retention_years": 15, "custodian": "PS-03"},
        "ST-04": {"title": "Flu Vaccine Response", "pi": "PS-04", "status": "closed", "closed_on": "2020-01-15",
                  "retention_years": 5, "custodian": "PS-05"},
        "ST-05": {"title": "Gut Microbiome Pilot", "pi": "PS-09", "status": "closed", "closed_on": "2022-12-01",
                  "retention_years": 7, "custodian": "PS-01"},
    }
    participants = dict((k, {"study": st, "consent": c, "withdrawn_on": w}) for k, st, c, w in [
        ("PT-101", "ST-01", "given", ""), ("PT-118", "ST-03", "given", ""), ("PT-120", "ST-01", "given", ""),
        ("PT-130", "ST-01", "given", ""), ("PT-150", "ST-01", "withdrawn", "2026-10-30"),
        ("PT-205", "ST-03", "given", ""), ("PT-210", "ST-03", "given", ""), ("PT-220", "ST-01", "given", ""),
        ("PT-301", "ST-02", "given", ""), ("PT-401", "ST-04", "given", ""), ("PT-501", "ST-05", "given", "")])
    freezers = dict((k, {"name": n, "temp": t, "room": r, "status": st}) for k, n, t, r, st in [
        ("FZ-01", "DNA freezer", -20, "B12", "ok"), ("FZ-02", "Freezer B", -80, "B14", "alarm"),
        ("FZ-03", "Freezer C", -80, "B14", "defrost"), ("FZ-04", "Freezer D", -80, "B15", "ok"),
        ("FZ-05", "Freezer E", -80, "B15", "ok"), ("FZ-06", "LN2 Dewar 1", -196, "B16", "ok"),
        ("FZ-07", "LN2 Dewar 2", -196, "B16", "ok")])
    racks = dict((k, {"freezer": f, "slots": n}) for k, f, n in [
        ("RK-11", "FZ-01", 20), ("RK-21", "FZ-02", 10), ("RK-22", "FZ-02", 10), ("RK-31", "FZ-03", 12),
        ("RK-32", "FZ-03", 12), ("RK-41", "FZ-04", 6), ("RK-42", "FZ-04", 6), ("RK-51", "FZ-05", 10),
        ("RK-52", "FZ-05", 10), ("RK-61", "FZ-06", 8), ("RK-71", "FZ-07", 8)])
    samples = {}
    for k, pt, typ, vol, coll, rack, slot, status, holder, purpose, due, hold in [
        ("SM-0301", "PT-301", "plasma", 400, "2007-03-02", "RK-51", 1, "stored", "", "", "", ""),
        ("SM-0302", "PT-301", "serum", 350, "2007-03-02", "RK-51", 2, "stored", "", "", "", ""),
        ("SM-0303", "PT-301", "tissue", 80, "2007-03-02", "RK-61", 1, "stored", "", "", "", ""),
        ("SM-0304", "PT-301", "plasma", 300, "2007-04-11", "RK-52", 2, "stored", "", "", "", "audit"),
        ("SM-0305", "PT-301", "dna", 120, "2007-04-11", "", 0, "destroyed", "", "", "", ""),
        ("SM-0412", "PT-101", "plasma", 1200, "2026-09-14", "RK-51", 3, "stored", "", "", "", ""),
        ("SM-0451", "PT-120", "plasma", 500, "2026-08-02", "RK-51", 7, "stored", "", "", "", ""),
        ("SM-0452", "PT-120", "serum", 500, "2026-08-02", "RK-51", 8, "stored", "", "", "", ""),
        ("SM-0453", "PT-120", "urine", 900, "2026-08-02", "RK-51", 10, "stored", "", "", "", "quality"),
        ("SM-0461", "PT-401", "dna", 150, "2019-01-20", "RK-11", 1, "stored", "", "", "", ""),
        ("SM-0462", "PT-401", "dna", 150, "2019-02-20", "RK-11", 2, "stored", "", "", "", ""),
        ("SM-0501", "PT-501", "plasma", 250, "2022-05-05", "RK-52", 4, "stored", "", "", "", ""),
        ("SM-0520", "PT-101", "serum", 600, "2026-09-14", "RK-31", 1, "stored", "", "", "", ""),
        ("SM-0601", "PT-210", "rna", 200, "2026-07-01", "", 0, "checked_out", "PS-06", "assay", "2026-11-13", ""),
        ("SM-0602", "PT-210", "rna", 180, "2026-07-01", "", 0, "checked_out", "PS-06", "assay", "2026-11-05", ""),
        ("SM-0603", "PT-210", "cells", 1000, "2026-07-01", "", 0, "checked_out", "PS-06", "assay", "2026-11-09", ""),
        ("SM-0604", "PT-210", "dna", 90, "2026-07-01", "", 0, "checked_out", "PS-06", "quality_check", "2026-11-10", ""),
        ("SM-0701", "PT-220", "plasma", 450, "2026-10-01", "", 0, "checked_out", "PS-07", "assay", "2026-11-20", ""),
        ("SM-0702", "PT-220", "serum", 450, "2026-10-01", "", 0, "checked_out", "PS-07", "assay", "2026-11-20", ""),
        ("SM-0703", "PT-220", "tissue", 60, "2026-10-01", "", 0, "checked_out", "PS-07", "aliquoting", "2026-11-18", ""),
        ("SM-0704", "PT-220", "dna", 100, "2026-10-01", "RK-11", 5, "stored", "", "", "", ""),
        ("SM-0705", "PT-220", "dna", 100, "2026-10-01", "", 0, "checked_out", "PS-07", "assay", "2026-11-19", ""),
        ("SM-0801", "PT-150", "plasma", 500, "2026-06-10", "RK-32", 3, "stored", "", "", "", ""),
        ("SM-0802", "PT-150", "serum", 500, "2026-06-10", "RK-32", 4, "stored", "", "", "", ""),
        ("SM-0901", "PT-118", "plasma", 700, "2026-05-22", "RK-32", 5, "stored", "", "", "", ""),
        ("SM-1001", "PT-130", "plasma", 400, "2026-04-04", "RK-22", 2, "stored", "", "", "", ""),
        ("SM-1002", "PT-130", "serum", 400, "2026-04-04", "RK-21", 1, "stored", "", "", "", ""),
        ("SM-1003", "PT-130", "rna", 150, "2026-04-04", "RK-21", 5, "stored", "", "", "", ""),
        ("SM-1004", "PT-130", "plasma", 400, "2026-05-04", "RK-22", 6, "stored", "", "", "", "quality"),
        ("SM-1101", "PT-205", "plasma", 500, "2026-02-14", "RK-32", 1, "stored", "", "", "", ""),
        ("SM-1102", "PT-205", "serum", 500, "2026-02-14", "RK-32", 2, "stored", "", "", "", ""),
        ("SM-1103", "PT-205", "tissue", 50, "2026-02-14", "RK-61", 2, "stored", "", "", "", ""),
        ("SM-1104", "PT-205", "dna", 80, "2026-02-14", "RK-11", 3, "stored", "", "", "", "legal"),
        ("SM-1201", "PT-101", "plasma", 450, "2026-10-20", "RK-52", 1, "stored", "", "", "", ""),
        ("SM-1301", "PT-101", "plasma", 300, "2026-03-03", "RK-41", 1, "stored", "", "", "", ""),
        ("SM-1302", "PT-101", "serum", 300, "2026-03-03", "RK-41", 2, "stored", "", "", "", ""),
        ("SM-1303", "PT-120", "urine", 800, "2026-03-03", "RK-41", 4, "stored", "", "", "", ""),
        ("SM-1304", "PT-120", "plasma", 300, "2026-03-03", "RK-42", 1, "stored", "", "", "", ""),
        ("SM-1305", "PT-130", "serum", 250, "2026-03-03", "", 0, "destroyed", "", "", "", ""),
    ]:
        study = participants[pt]["study"]
        samples[k] = {"study": study, "participant": pt, "type": typ, "volume_ul": vol, "collected": coll,
                      "status": status, "rack": rack if status == "stored" else "",
                      "slot": slot if status == "stored" else 0,
                      "custodian": holder or studies[study]["custodian"], "hold": hold, "hold_review": "",
                      "parent": "", "purpose": purpose, "due_back": due,
                      "destroyed_on": "2025-02-01" if status == "destroyed" else "",
                      "destroy_method": "autoclave" if status == "destroyed" else "",
                      "witness": "PS-03" if status == "destroyed" else ""}
    samples["SM-0305"].update({"destroy_method": "incineration", "witness": "PS-05"})
    st = {"today": "2026-11-11",
          "policy": {"storage_temp": {"plasma": -80, "serum": -80, "rna": -80, "urine": -80, "dna": -20,
                                      "tissue": -196, "cells": -196},
                     "methods": {"plasma": ["autoclave", "incineration"], "serum": ["autoclave", "incineration"],
                                 "rna": ["autoclave", "incineration"], "urine": ["autoclave", "chemical"],
                                 "dna": ["chemical", "incineration"], "tissue": ["incineration"],
                                 "cells": ["incineration"]},
                     "max_checkout_days": 14, "min_parent_ul": 100, "min_aliquot_ul": 20},
          "staff": staff, "studies": studies, "participants": participants, "freezers": freezers, "racks": racks,
          "samples": samples,
          "custody_log": {"CT-700": {"samples": ["SM-0601", "SM-0602"], "from": "PS-03", "to": "PS-06",
                                     "date": "2026-10-29", "reason": "check_out"},
                          "CT-701": {"samples": ["SM-0701", "SM-0702", "SM-0703", "SM-0705"], "from": "PS-03",
                                     "to": "PS-07", "date": "2026-11-04", "reason": "check_out"}},
          "next_sample": 2000, "next_custody": 800}
    seen = set()
    for k, s in samples.items():   # one sample per slot, right temperature
        if s["status"] == "stored":
            assert (s["rack"], s["slot"]) not in seen, k
            seen.add((s["rack"], s["slot"]))
            assert freezers[racks[s["rack"]]["freezer"]]["temp"] == st["policy"]["storage_temp"][s["type"]], k
            assert s["slot"] <= racks[s["rack"]]["slots"], k
    return st


def _smp(parent_rec, parent, vol, rack, slot, custodian):
    return {"study": parent_rec["study"], "participant": parent_rec["participant"], "type": parent_rec["type"],
            "volume_ul": vol, "collected": parent_rec["collected"], "status": "stored", "rack": rack, "slot": slot,
            "custodian": custodian, "hold": "", "hold_review": "", "parent": parent, "purpose": "", "due_back": "",
            "destroyed_on": "", "destroy_method": "", "witness": ""}


def _ct(samples, frm, to, reason, date="2026-11-11"):
    return {"samples": sorted(samples), "from": frm, "to": to, "date": date, "reason": reason}


def _destroyed(method, witness, date="2026-11-11"):
    return {"status": "destroyed", "rack": "", "slot": 0, "destroyed_on": date, "destroy_method": method,
            "witness": witness}


_LT = "2026-11-11"   # lab "today"
_LAB0 = _lab()

# ---- single-call S13-S18
_add("S13", "lab", _lab(),
     "Please split plasma sample SM-0412 (1.2 mL left, collected 14 September, cohort ST-01) into three aliquots "
     "of 0.25 mL each for the troponin assay. Put them in rack RK-51, slots 4, 5 and 6 - the parent itself sits "
     "in slot 3 of the same rack and stays there.",
     [["create_aliquots", {"sample_id": "SM-0412", "aliquots": [
         {"volume_ul": 250, "rack_id": "RK-51", "slot": 4}, {"volume_ul": 250, "rack_id": "RK-51", "slot": 5},
         {"volume_ul": 250, "rack_id": "RK-51", "slot": 6}]}]],
     [["create_aliquots", {"sample_id": "SM-0412", "aliquots": [
         {"volume_ul": 25, "rack_id": "RK-51", "slot": 4}, {"volume_ul": 25, "rack_id": "RK-51", "slot": 5},
         {"volume_ul": 25, "rack_id": "RK-51", "slot": 6}]}]],
     lambda s, i: _final(s, i, changes={"samples": {"SM-0412": {"volume_ul": 450}}}, new={"samples": [
         _smp(_LAB0["samples"]["SM-0412"], "SM-0412", 250, "RK-51", n, "PS-03") for n in (4, 5, 6)]}),
     "OUX")

_add("S14", "lab", _lab(),
     "Dr Mira Patel (PS-04) needs serum sample SM-0520 for an immunoassay run in her lab - she is taking it out to run "
     "the assay. She will bring it back one week from today; today is Wednesday 11 November 2026. Record that "
     "she now has it.",
     [["check_out_sample", {"sample_id": "SM-0520", "to_person": "PS-04", "purpose": "assay",
                            "due_back": "2026-11-18"}]],
     [["transfer_custody", {"sample_ids": ["SM-0520"], "from_person": "PS-03", "to_person": "PS-04",
                            "reason": "handover"}],
      ["check_out_sample", {"sample_id": "SM-0520", "to_person": "PS-04", "purpose": "aliquoting",
                            "due_back": "2026-11-18"}]],
     lambda s, i: _final(s, i, changes={"samples": {"SM-0520": {
         "status": "checked_out", "rack": "", "slot": 0, "custodian": "PS-04", "purpose": "assay",
         "due_back": "2026-11-18"}}}, new={"custody_log": [_ct(["SM-0520"], "PS-03", "PS-04", "check_out")]}),
     "TND", mentions=["Mira Patel"])

_add("S15", "lab", _lab(),
     "Leo Martins (PS-06) has finished the qPCR with RNA sample SM-0601 and is handing it back. He used 35 uL of "
     "the 200 uL. Put it back into storage in rack RK-51, slot 9, and record the volume he used.",
     [["return_sample", {"sample_id": "SM-0601", "rack_id": "RK-51", "slot": 9, "volume_used_ul": 35}]],
     [["return_sample", {"sample_id": "SM-0601", "rack_id": "RK-51", "slot": 9}]],
     lambda s, i: _final(s, i, changes={"samples": {"SM-0601": {
         "status": "stored", "rack": "RK-51", "slot": 9, "custodian": "PS-03", "purpose": "", "due_back": "",
         "volume_ul": 165}}}, new={"custody_log": [_ct(["SM-0601"], "PS-06", "PS-03", "return")]}),
     "TPX", mentions=["Leo Martins"])

_add("S16", "lab", _lab(),
     "Shift change on the bench: Anika Rao (PS-07) is going off shift and hands three of the samples she has out "
     "to Tomas Ruiz (PS-08): SM-0701, SM-0702 and SM-0705. She keeps the biopsy SM-0703 herself, and SM-0704 is "
     "already back in the freezer. Record the hand-over.",
     [["transfer_custody", {"sample_ids": ["SM-0701", "SM-0702", "SM-0705"], "from_person": "PS-07",
                            "to_person": "PS-08", "reason": "shift_change"}]],
     [["transfer_custody", {"sample_ids": ["SM-0701", "SM-0702", "SM-0705"], "from_person": "PS-07",
                            "to_person": "PS-08", "reason": "handover"}]],
     lambda s, i: _final(s, i, changes={"samples": dict((k, {"custodian": "PS-08"}) for k in
                                                        ("SM-0701", "SM-0702", "SM-0705"))},
                         new={"custody_log": [_ct(["SM-0701", "SM-0702", "SM-0705"], "PS-07", "PS-08",
                                                  "shift_change")]}),
     "NXT", mentions=["Anika Rao", "Tomas Ruiz"])

_add("S17", "lab", _lab(),
     "Participant PT-150 withdrew consent on 30 October, so their plasma SM-0801 and serum SM-0802 have to be "
     "destroyed. Both go through steam sterilisation in the pressure steriliser this afternoon. Jonas Weber "
     "(PS-05) is the witness. Record the destruction.",
     [["destroy_samples", {"sample_ids": ["SM-0801", "SM-0802"], "method": "autoclave", "witness_id": "PS-05"}]],
     [["destroy_samples", {"sample_ids": ["SM-0801", "SM-0802"], "method": "chemical", "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-0801", "SM-0802"], "method": "incineration", "witness_id": "PS-05"}]],
     lambda s, i: _final(s, i, changes={"samples": {"SM-0801": _destroyed("autoclave", "PS-05"),
                                                    "SM-0802": _destroyed("autoclave", "PS-05")}}),
     "NX", mentions=["Jonas Weber"])

_add("S18", "lab", _lab(),
     "Legal has told us a court case involving participant PT-118 is pending and all of their material must be "
     "preserved until it is resolved. Put plasma sample SM-0901 under the matching kind of hold, to be reviewed "
     "on 1 March 2027.",
     [["set_sample_hold", {"sample_id": "SM-0901", "on_hold": True, "reason": "legal", "review_on": "2027-03-01"}]],
     [["set_sample_hold", {"sample_id": "SM-0901", "on_hold": True, "reason": "audit", "review_on": "2027-03-01"}]],
     lambda s, i: _final(s, i, changes={"samples": {"SM-0901": {"hold": "legal", "hold_review": "2027-03-01"}}}),
     "NPD")

# ---- multi-step M13-M18
_add("M13", "lab", _lab(),
     "A letter arrived today: participant PT-205 has withdrawn consent. Record the withdrawal, then destroy every "
     "one of their samples that the rules allow to be destroyed now. Use autoclaving wherever it is permitted for "
     "the sample type and incineration otherwise. The witness for each destruction is the lowest-numbered staff "
     "member who is allowed to witness it. Nothing else should change.",
     [["get_lab_policy", {}], ["list_staff", {}], ["find_samples", {"participant_id": "PT-205"}],
      ["record_consent_withdrawal", {"participant_id": "PT-205", "date_received": "2026-11-11"}],
      ["destroy_samples", {"sample_ids": ["SM-1101", "SM-1102", "SM-1104"], "method": "autoclave",
                           "witness_id": "PS-01"}],
      ["destroy_samples", {"sample_ids": ["SM-1101", "SM-1102"], "method": "autoclave", "witness_id": "PS-01"}],
      ["destroy_samples", {"sample_ids": ["SM-1103"], "method": "incineration", "witness_id": "PS-01"}],
      ["find_samples", {"participant_id": "PT-205"}]],
     [["record_consent_withdrawal", {"participant_id": "PT-205", "date_received": "2026-11-11"}],
      ["destroy_samples", {"sample_ids": ["SM-1101", "SM-1102"], "method": "incineration", "witness_id": "PS-03"}],
      ["destroy_samples", {"sample_ids": ["SM-1101", "SM-1102"], "method": "incineration", "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-1103"], "method": "incineration", "witness_id": "PS-05"}]],
     lambda s, i: _final(s, i, changes={
         "participants": {"PT-205": {"consent": "withdrawn", "withdrawn_on": _LT}},
         "samples": {"SM-1101": _destroyed("autoclave", "PS-01"), "SM-1102": _destroyed("autoclave", "PS-01"),
                     "SM-1103": _destroyed("incineration", "PS-01")}}),
     "abcf", ref_errors=[4])

_add("M14", "lab", _lab(),
     "Freezer FZ-02 is alarming and must be emptied. Move every sample stored in it, in order of sample id, to "
     "the lowest-numbered freezer that can take it right now, each into the lowest free slot of that freezer's "
     "lowest-numbered rack that still has one. Samples on hold are moved too. Nothing else should change.",
     [["get_lab_policy", {}], ["list_storage", {}], ["list_storage", {"freezer_id": "FZ-02"}],
      ["relocate_sample", {"sample_id": "SM-1001", "rack_id": "RK-31", "slot": 2}],
      ["list_storage", {"freezer_id": "FZ-04"}],
      ["relocate_sample", {"sample_id": "SM-1001", "rack_id": "RK-41", "slot": 3}],
      ["relocate_sample", {"sample_id": "SM-1002", "rack_id": "RK-41", "slot": 5}],
      ["relocate_sample", {"sample_id": "SM-1003", "rack_id": "RK-41", "slot": 6}],
      ["relocate_sample", {"sample_id": "SM-1004", "rack_id": "RK-42", "slot": 2}],
      ["list_storage", {"freezer_id": "FZ-02"}]],
     [["relocate_sample", {"sample_id": "SM-1001", "rack_id": "RK-41", "slot": 3}],
      ["relocate_sample", {"sample_id": "SM-1002", "rack_id": "RK-41", "slot": 5}],
      ["relocate_sample", {"sample_id": "SM-1003", "rack_id": "RK-41", "slot": 6}]],
     lambda s, i: _final(s, i, changes={"samples": {
         "SM-1001": {"rack": "RK-41", "slot": 3}, "SM-1002": {"rack": "RK-41", "slot": 5},
         "SM-1003": {"rack": "RK-41", "slot": 6}, "SM-1004": {"rack": "RK-42", "slot": 2}}}),
     "abdf", ref_errors=[3])

_add("M15", "lab", _lab(),
     "Samples SM-0451, SM-0452 and SM-0453 are going to our partner lab. Check each of them out to Ravi Shah for "
     "shipment preparation, due back two weeks from today. Any of them that may not be released stays where it "
     "is. Then Ravi hands everything he took out for this over to the courier: the lowest-numbered member of "
     "staff who is allowed to take it as a courier. Nothing else should change.",
     [["get_lab_policy", {}], ["list_staff", {}],
      ["check_out_sample", {"sample_id": "SM-0451", "to_person": "PS-09", "purpose": "shipment_prep",
                            "due_back": "2026-11-25"}],
      ["check_out_sample", {"sample_id": "SM-0452", "to_person": "PS-09", "purpose": "shipment_prep",
                            "due_back": "2026-11-25"}],
      ["check_out_sample", {"sample_id": "SM-0453", "to_person": "PS-09", "purpose": "shipment_prep",
                            "due_back": "2026-11-25"}],
      ["find_samples", {"custodian_id": "PS-09"}],
      ["transfer_custody", {"sample_ids": ["SM-0451", "SM-0452"], "from_person": "PS-09", "to_person": "PS-10",
                            "reason": "courier"}]],
     [["check_out_sample", {"sample_id": "SM-0451", "to_person": "PS-09", "purpose": "shipment_prep",
                            "due_back": "2026-11-25"}],
      ["check_out_sample", {"sample_id": "SM-0452", "to_person": "PS-09", "purpose": "shipment_prep",
                            "due_back": "2026-11-25"}],
      ["transfer_custody", {"sample_ids": ["SM-0451", "SM-0452"], "from_person": "PS-09", "to_person": "PS-11",
                            "reason": "courier"}]],
     lambda s, i: _final(s, i, changes={"samples": dict(
         (k, {"status": "checked_out", "rack": "", "slot": 0, "custodian": "PS-10", "purpose": "shipment_prep",
              "due_back": "2026-11-25"}) for k in ("SM-0451", "SM-0452"))},
         new={"custody_log": [_ct(["SM-0451"], "PS-03", "PS-09", "check_out"),
                              _ct(["SM-0452"], "PS-03", "PS-09", "check_out"),
                              _ct(["SM-0451", "SM-0452"], "PS-09", "PS-10", "courier")]}),
     "abcef", ref_errors=[4], mentions=["Ravi Shah"], seq=[[2, 6], [3, 6]])

_add("M16", "lab", _lab(),
     "Annual retention review. Destroy every stored sample from a closed study whose retention period is over, "
     "unless it is on hold. Use autoclaving wherever it is permitted for the sample type and incineration "
     "otherwise. Jonas Weber witnesses every destruction he is allowed to witness; where he isn't, Kim Tran "
     "witnesses instead. Studies still in retention and active studies are not touched, and nothing else "
     "should change.",
     [["get_lab_policy", {}], ["list_studies", {}],
      ["find_samples", {"study_id": "ST-02", "status": "stored"}],
      ["find_samples", {"study_id": "ST-04", "status": "stored"}],
      ["destroy_samples", {"sample_ids": ["SM-0301", "SM-0302", "SM-0304"], "method": "autoclave",
                           "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-0301", "SM-0302"], "method": "autoclave", "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-0303"], "method": "incineration", "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-0461", "SM-0462"], "method": "incineration", "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-0461", "SM-0462"], "method": "incineration", "witness_id": "PS-03"}],
      ["find_samples", {"study_id": "ST-02", "status": "stored"}]],
     [["destroy_samples", {"sample_ids": ["SM-0301", "SM-0302"], "method": "autoclave", "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-0303"], "method": "incineration", "witness_id": "PS-05"}],
      ["destroy_samples", {"sample_ids": ["SM-0461", "SM-0462"], "method": "chemical", "witness_id": "PS-03"}],
      ["destroy_samples", {"sample_ids": ["SM-0501"], "method": "autoclave", "witness_id": "PS-05"}]],
     lambda s, i: _final(s, i, changes={"samples": {
         "SM-0301": _destroyed("autoclave", "PS-05"), "SM-0302": _destroyed("autoclave", "PS-05"),
         "SM-0303": _destroyed("incineration", "PS-05"), "SM-0461": _destroyed("incineration", "PS-03"),
         "SM-0462": _destroyed("incineration", "PS-03")}}),
     "abdf", ref_errors=[4, 7], mentions=["Jonas Weber", "Kim Tran"])

_add("M17", "lab", _lab(),
     "For the new assay, make as many 100 uL aliquots of plasma sample SM-1201 as the rules allow, up to four, "
     "into the lowest free slots of the rack the parent is stored in. Then check the aliquot with the lowest "
     "sample id out to Dr Mira Patel for the assay, due back on 20 November. Nothing else should change.",
     [["get_lab_policy", {}], ["get_sample", {"sample_id": "SM-1201"}], ["list_storage", {"freezer_id": "FZ-05"}],
      ["create_aliquots", {"sample_id": "SM-1201", "aliquots": [
          {"volume_ul": 100, "rack_id": "RK-52", "slot": 3}, {"volume_ul": 100, "rack_id": "RK-52", "slot": 5},
          {"volume_ul": 100, "rack_id": "RK-52", "slot": 6}, {"volume_ul": 100, "rack_id": "RK-52", "slot": 7}]}],
      ["create_aliquots", {"sample_id": "SM-1201", "aliquots": [
          {"volume_ul": 100, "rack_id": "RK-52", "slot": 3}, {"volume_ul": 100, "rack_id": "RK-52", "slot": 5},
          {"volume_ul": 100, "rack_id": "RK-52", "slot": 6}]}],
      ["get_sample", {"sample_id": "SM-1201"}],
      ["check_out_sample", {"sample_id": "SM-2000", "to_person": "PS-04", "purpose": "assay",
                            "due_back": "2026-11-20"}]],
     [["create_aliquots", {"sample_id": "SM-1201", "aliquots": [
         {"volume_ul": 100, "rack_id": "RK-52", "slot": 3}, {"volume_ul": 100, "rack_id": "RK-52", "slot": 5}]}],
      ["check_out_sample", {"sample_id": "SM-2000", "to_person": "PS-04", "purpose": "assay",
                            "due_back": "2026-11-20"}]],
     _any(*[({"samples": {"SM-1201": {"volume_ul": 150}}}, {
         "samples": [dict(_smp(_LAB0["samples"]["SM-1201"], "SM-1201", 100, "", 0, "PS-04"),
                          status="checked_out", purpose="assay", due_back="2026-11-20")] +
                    [_smp(_LAB0["samples"]["SM-1201"], "SM-1201", 100, "RK-52", n, "PS-03") for n in (3, 5, 6) if n != out],
         "custody_log": [_ct(["SM-2000"], "PS-03", "PS-04", "check_out")]}) for out in (3, 5, 6)]),
     "abcef", ref_errors=[3], mentions=["Mira Patel"])

_add("M18", "lab", _lab(),
     "Anika Rao is leaving the lab today. Hand every sample she currently has checked out to her supervisor. Any "
     "of them her supervisor isn't allowed to hold goes instead to the lowest-numbered other active staff member "
     "who is. Record these as hand-overs. Samples she has already returned are not part of this, and nothing "
     "else should change.",
     [["get_lab_policy", {}], ["list_staff", {}], ["find_samples", {"custodian_id": "PS-07"}],
      ["transfer_custody", {"sample_ids": ["SM-0701", "SM-0702", "SM-0703", "SM-0705"], "from_person": "PS-07",
                            "to_person": "PS-04", "reason": "handover"}],
      ["transfer_custody", {"sample_ids": ["SM-0701", "SM-0702", "SM-0705"], "from_person": "PS-07",
                            "to_person": "PS-04", "reason": "handover"}],
      ["transfer_custody", {"sample_ids": ["SM-0703"], "from_person": "PS-07", "to_person": "PS-01",
                            "reason": "handover"}],
      ["find_samples", {"custodian_id": "PS-07"}]],
     [["transfer_custody", {"sample_ids": ["SM-0701", "SM-0702", "SM-0705"], "from_person": "PS-07",
                            "to_person": "PS-04", "reason": "handover"}],
      ["transfer_custody", {"sample_ids": ["SM-0703"], "from_person": "PS-07", "to_person": "PS-03",
                            "reason": "handover"}]],
     lambda s, i: _final(s, i, changes={"samples": {
         "SM-0701": {"custodian": "PS-04"}, "SM-0702": {"custodian": "PS-04"}, "SM-0705": {"custodian": "PS-04"},
         "SM-0703": {"custodian": "PS-01"}}}, new={"custody_log": [
             _ct(["SM-0701", "SM-0702", "SM-0705"], "PS-07", "PS-04", "handover"),
             _ct(["SM-0703"], "PS-07", "PS-01", "handover")]}),
     "abdf", ref_errors=[3], mentions=["Anika Rao"])

# ---- calibration C3 (single), C8 (multi)
_add("C3", "lab", _lab(),
     "Freezer B (FZ-02) keeps alarming, so move plasma sample SM-1001 out of it now: it is in rack RK-22, slot 2. "
     "Its new place is rack RK-52 in Freezer E, slot 7 (not slot 2, which is taken). It stays in storage; "
     "nobody is taking it out.",
     [["relocate_sample", {"sample_id": "SM-1001", "rack_id": "RK-52", "slot": 7}]],
     [["check_out_sample", {"sample_id": "SM-1001", "to_person": "PS-03", "purpose": "quality_check",
                            "due_back": "2026-11-12"}],
      ["return_sample", {"sample_id": "SM-1001", "rack_id": "RK-52", "slot": 7}]],
     lambda s, i: _final(s, i, changes={"samples": {"SM-1001": {"rack": "RK-52", "slot": 7}}}),
     "TX")

_add("C8", "lab", _lab(),
     "Leo Martins has brought back every sample he had checked out whose due-back date has already passed. Put "
     "each one back into storage, in order of sample id, in the lowest-numbered freezer that can take it right "
     "now, in the lowest free slot of that freezer's lowest-numbered rack that has one. He didn't note any "
     "volume used. Samples he still has that aren't overdue stay with him, and nothing else should change.",
     [["get_lab_policy", {}], ["find_samples", {"custodian_id": "PS-06", "status": "checked_out"}],
      ["list_storage", {}],
      ["return_sample", {"sample_id": "SM-0602", "rack_id": "RK-21", "slot": 2}],
      ["list_storage", {"freezer_id": "FZ-04"}],
      ["return_sample", {"sample_id": "SM-0602", "rack_id": "RK-41", "slot": 3}],
      ["list_storage", {"freezer_id": "FZ-06"}],
      ["return_sample", {"sample_id": "SM-0603", "rack_id": "RK-61", "slot": 3}],
      ["list_storage", {"freezer_id": "FZ-01"}],
      ["return_sample", {"sample_id": "SM-0604", "rack_id": "RK-11", "slot": 4}],
      ["find_samples", {"custodian_id": "PS-06", "status": "checked_out"}]],
     [["return_sample", {"sample_id": "SM-0601", "rack_id": "RK-41", "slot": 3}],
      ["return_sample", {"sample_id": "SM-0602", "rack_id": "RK-41", "slot": 5}],
      ["return_sample", {"sample_id": "SM-0603", "rack_id": "RK-61", "slot": 3}],
      ["return_sample", {"sample_id": "SM-0604", "rack_id": "RK-11", "slot": 4}]],
     lambda s, i: _final(s, i, changes={"samples": {
         "SM-0602": {"status": "stored", "rack": "RK-41", "slot": 3, "custodian": "PS-03", "purpose": "", "due_back": ""},
         "SM-0603": {"status": "stored", "rack": "RK-61", "slot": 3, "custodian": "PS-03", "purpose": "", "due_back": ""},
         "SM-0604": {"status": "stored", "rack": "RK-11", "slot": 4, "custodian": "PS-03", "purpose": "",
                     "due_back": ""}}},
         new={"custody_log": [_ct(["SM-0602"], "PS-06", "PS-03", "return"), _ct(["SM-0603"], "PS-06", "PS-03", "return"),
                              _ct(["SM-0604"], "PS-06", "PS-03", "return")]}),
     "abdf", ref_errors=[3], mentions=["Leo Martins"])


# --------------------------------------------------------------------------
# 4. community sports league: fixtures, venues, referees, team sheets, eligibility and suspensions
# --------------------------------------------------------------------------
_DIVISIONS = ["senior", "u14", "u12"]
_FIX_STATUS = ["scheduled", "postponed", "played", "cancelled"]
_POSTPONE_REASONS = ["bad_weather", "pitch_unfit", "referee_unavailable", "team_request"]
_ROLES = ["referee", "assistant_1", "assistant_2"]
_SUSP_REASONS = ["red_card", "accumulated_yellows", "misconduct"]
_POSITIONS = ["goalkeeper", "defender", "midfielder", "forward"]


class LeagueEnv(_Env):
    TOOLS = [
        _fn("get_league_rules", "Today's date, the divisions (side size, squad size, match length, minimum referee "
            "grade, age limits) and the league rules that the write tools enforce."),
        _fn("list_fixtures", "Fixtures, filtered, by date and kick-off; 8 per page. Shows venue, pitch, officials, "
            "result, cards and submitted team sheets.",
            {"date": _s("Only this date, YYYY-MM-DD."), "team_id": _s("Only fixtures of this team, e.g. TM-21."),
             "venue_id": _s("Only this venue, e.g. VN-2."), "status": _s("Only this status.", _FIX_STATUS),
             "page": _PAGE}),
        _fn("list_venues", "Venues: pitches, floodlit pitches, opening hours and closed dates."),
        _fn("list_referees", "Referees: grade, club they belong to, weekdays they are available, dates they are "
            "unavailable, and their appointments on a given date.",
            {"available_on": _s("Only referees available on this date, YYYY-MM-DD (also lists their appointments "
                                "that day)."),
             "min_grade": _i("Only referees with at least this grade.", 1, 5)}),
        _fn("find_players", "Search players by name and/or team; shows registration date, season cards and status.",
            {"query": _s("Text that must appear in the name."), "team_id": _s("Only this team.")}),
        _fn("get_player", "One player: team, date of birth, registration, season cards, suspensions and the team's "
            "next fixtures.", {"player_id": _s("Player id, e.g. PL-604.")}, ["player_id"]),
        _fn("schedule_fixture", "Create a new fixture between two teams of the same division.",
            {"home_team": _s("Home team id."), "away_team": _s("Away team id."), "date": _s("Date, YYYY-MM-DD."),
             "kickoff": _s("Kick-off time HH:MM (15-minute steps)."), "venue_id": _s("Venue id."),
             "pitch": _i("Pitch number at the venue.", 1, 10)},
            ["home_team", "away_team", "date", "kickoff", "venue_id", "pitch"]),
        _fn("reschedule_fixture", "Move a scheduled or postponed fixture to a new date/kick-off and optionally a new "
            "venue and pitch; it becomes scheduled. Changing the date or kick-off removes its officials.",
            {"fixture_id": _s("Fixture id, e.g. FX-214."), "date": _s("New date, YYYY-MM-DD."),
             "kickoff": _s("New kick-off HH:MM."), "venue_id": _s("New venue (default: unchanged)."),
             "pitch": _i("New pitch (default: unchanged).", 1, 10)}, ["fixture_id", "date", "kickoff"]),
        _fn("postpone_fixture", "Postpone a scheduled fixture (no new date yet); its officials are removed.",
            {"fixture_id": _s("Fixture id."), "reason": _s("Why.", _POSTPONE_REASONS)}, ["fixture_id", "reason"]),
        _fn("assign_referee", "Appoint a referee to a role on a scheduled fixture (replaces whoever held that role).",
            {"fixture_id": _s("Fixture id."), "referee_id": _s("Referee id, e.g. RF-02."),
             "role": _s("Role.", _ROLES)}, ["fixture_id", "referee_id", "role"]),
        _fn("record_result", "Record the final score and the cards of a fixture that has been played.",
            {"fixture_id": _s("Fixture id."), "home_goals": _i("Home goals.", 0, 30), "away_goals": _i("Away goals.", 0, 30),
             "cards": _a("Cards shown (optional).", _o({"player_id": _s("Player id."),
                                                        "card": _s("Card colour.", ["yellow", "red"]),
                                                        "minute": _i("Match minute (stoppage time added on, e.g. "
                                                                     "45+2 is 47).", 1, 130)},
                                                       ["player_id", "card", "minute"]), 1, 20)},
            ["fixture_id", "home_goals", "away_goals"]),
        _fn("issue_suspension", "Suspend a player for a number of their team's matches.",
            {"player_id": _s("Player id."), "matches": _i("Number of matches.", 1, 10),
             "reason": _s("Why.", _SUSP_REASONS)}, ["player_id", "matches", "reason"]),
        _fn("submit_team_sheet", "Submit (or replace) a team's sheet for a fixture: every player with shirt number "
            "and whether they start, plus the captain.",
            {"fixture_id": _s("Fixture id."), "team_id": _s("Team id."),
             "players": _a("The players named.", _o({"player_id": _s("Player id."),
                                                     "shirt": _i("Shirt number.", 1, 99),
                                                     "starting": _b("Starts the match (default true); false for "
                                                                    "substitutes.")},
                                                    ["player_id", "shirt"]), 1, 20),
             "captain_id": _s("Player id of the captain.")},
            ["fixture_id", "team_id", "players", "captain_id"]),
        _fn("register_player", "Register a new player for a team (registered today).",
            {"name": _s("Full name."), "team_id": _s("Team id."), "dob": _s("Date of birth, YYYY-MM-DD."),
             "position": _s("Optional usual position.", _POSITIONS)}, ["name", "team_id", "dob"]),
    ]

    def _fix(self, fid):
        return self._get("fixtures", fid, "fixture")

    def _team(self, tid):
        return self._get("teams", tid, "team")

    def _div(self, f):
        return self.state["policy"]["divisions"][f["division"]]

    def _end(self, f):
        return _ptime(f["kickoff"]) + self._div(f)["match_minutes"]

    def _team_name(self, tid):
        return self.state["teams"][tid]["name"]

    def _yellows(self, pid):
        n = self.state["players"][pid]["prior_yellows"]
        for f in self.state["fixtures"].values():
            if f["status"] == "played":
                n += len([c for c in f["cards"] if c["player_id"] == pid and c["card"] == "yellow"])
        return n

    def _reds(self, pid):
        return len([c for f in self.state["fixtures"].values() if f["status"] == "played"
                    for c in f["cards"] if c["player_id"] == pid and c["card"] == "red"])

    def _fix_row(self, k):
        f = self.state["fixtures"][k]
        sheets = [dict(sheet_id=sk, **s) for sk, s in sorted(self.state["team_sheets"].items()) if s["fixture"] == k]
        return {"fixture_id": k, "division": f["division"], "home_team": f["home"], "home": self._team_name(f["home"]),
                "away_team": f["away"], "away": self._team_name(f["away"]), "date": f["date"], "kickoff": f["kickoff"],
                "ends": _fmt_time(self._end(f)), "venue_id": f["venue"], "pitch": f["pitch"], "status": f["status"],
                "referee": f["referee"], "assistant_1": f["assistant_1"], "assistant_2": f["assistant_2"],
                "postpone_reason": f["postpone_reason"], "result": f["result"], "cards": f["cards"],
                "team_sheets": sheets}

    def _t_get_league_rules(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "weekday": _DAYNAMES[self._today().weekday()],
                "divisions": p["divisions"], "turnaround_minutes": p["turnaround"], "dusk": p["dusk"],
                "registration_days": p["registration_days"], "red_card_matches": p["red_card_matches"],
                "yellow_limit": p["yellow_limit"], "yellow_ban_matches": p["yellow_ban_matches"],
                "rules": [
                    "Fixtures kick off on a 15-minute mark, must fit inside the venue's opening hours (kick-off "
                    "plus the division's match_minutes), cannot be on a date the venue is closed or in the past, and "
                    "any fixture ending after dusk needs a floodlit pitch.",
                    "Two fixtures on the same pitch need turnaround_minutes between the end of one and the start "
                    "of the next. A team plays at most one fixture per day. Both teams must be in the same division.",
                    "Referees: the referee role needs at least the division's min_referee_grade (assistants any "
                    "grade). A referee must be available that weekday and not on an unavailable date, may not "
                    "officiate a fixture involving a team of their own club, holds one role per fixture, at most "
                    "2 fixtures a day, and needs 30 minutes between the end of one fixture and the start of the "
                    "next at the same venue, 60 minutes at a different venue.",
                    "Team sheets: exactly side_size starters, at most squad_size players, distinct shirt numbers, "
                    "and a captain who starts. Every player must belong to the team, be active, have been "
                    "registered at least registration_days before the fixture date, meet the division's age "
                    "limits, and not be suspended for that fixture.",
                    "A suspension of N matches is served by the next N fixtures of the player's team (scheduled or "
                    "played; postponed ones do not count) dated after the day it was issued; the player is "
                    "suspended for those N fixtures.",
                    "A red card shown in a recorded result requires a red_card suspension of red_card_matches. "
                    "Reaching yellow_limit yellow cards in the season (and again at each further multiple) "
                    "requires an accumulated_yellows suspension of yellow_ban_matches. Season yellows include "
                    "prior_yellows carried over from cup matches.",
                    "Results can be recorded only for scheduled fixtures dated today or earlier; carded players "
                    "must belong to one of the two teams (and be on its team sheet if one was submitted).",
                ]}

    def _t_list_fixtures(self, a):
        if "date" in a:
            _pdate(a["date"])
        if "team_id" in a:
            self._team(a["team_id"])
        if "venue_id" in a:
            self._get("venues", a["venue_id"], "venue")
        rows = [k for _, _, k in sorted((f["date"], f["kickoff"], k) for k, f in self.state["fixtures"].items()
                                        if a.get("date", f["date"]) == f["date"]
                                        and a.get("team_id", f["home"]) in (f["home"], f["away"])
                                        and a.get("venue_id", f["venue"]) == f["venue"]
                                        and a.get("status", f["status"]) == f["status"])]
        meta, rows = _paged(rows, a.get("page", 1), 8)
        meta["fixtures"] = [self._fix_row(k) for k in rows]
        return meta

    def _t_list_venues(self, a):
        return {"venues": [dict(venue_id=k, **v) for k, v in sorted(self.state["venues"].items())]}

    def _available(self, r, date):
        return _DAY3[_pdate(date).weekday()] in r["days"] and date not in r["unavailable"]

    def _t_list_referees(self, a):
        out = []
        for k, r in sorted(self.state["referees"].items()):
            if r["grade"] < a.get("min_grade", 1):
                continue
            row = dict(referee_id=k, **r)
            if "available_on" in a:
                if not self._available(r, a["available_on"]):
                    continue
                row["appointments"] = [{"fixture_id": fk, "role": role, "venue_id": f["venue"],
                                        "kickoff": f["kickoff"], "ends": _fmt_time(self._end(f))}
                                       for fk, f in sorted(self.state["fixtures"].items())
                                       if f["date"] == a["available_on"] and f["status"] == "scheduled"
                                       for role in _ROLES if f[role] == k]
            out.append(row)
        return {"referees": out}

    def _player_row(self, k):
        p = self.state["players"][k]
        return {"player_id": k, "name": p["name"], "team_id": p["team"], "team": self._team_name(p["team"]),
                "dob": p["dob"], "registered_on": p["registered_on"], "position": p["position"], "status": p["status"],
                "season_yellows": self._yellows(k), "red_cards": self._reds(k)}

    def _t_find_players(self, a):
        if "team_id" in a:
            self._team(a["team_id"])
        q = a.get("query", "").strip().lower()
        if "query" in a and not q:
            raise _Rule("query must not be empty")
        if not q and "team_id" not in a:
            raise _Rule("give a query or a team_id")
        return {"players": [self._player_row(k) for k, p in sorted(self.state["players"].items())
                            if q in p["name"].lower() and a.get("team_id", p["team"]) == p["team"]]}

    def _served(self, s, team, before=None):
        """Fixtures of the team that count towards suspension s, optionally only those before fixture key `before`."""
        out = []
        for _, _, k in sorted((f["date"], f["kickoff"], k) for k, f in self.state["fixtures"].items()
                              if team in (f["home"], f["away"]) and f["status"] in ("scheduled", "played")
                              and f["date"] > s["issued_on"]):
            if before is not None and k == before:
                break
            out.append(k)
        return out

    def _suspended_for(self, pid, fk):
        p = self.state["players"][pid]
        for sk, s in sorted(self.state["suspensions"].items()):
            if s["player"] != pid:
                continue
            f = self.state["fixtures"][fk]
            if f["date"] > s["issued_on"] and len(self._served(s, p["team"], before=fk)) < s["matches"]:
                return sk
        return ""

    def _t_get_player(self, a):
        k = a["player_id"]
        p = self._get("players", k, "player")
        row = self._player_row(k)
        row["eligible_from"] = _plus_days(p["registered_on"], self.state["policy"]["registration_days"])
        row["suspensions"] = [dict(suspension_id=sk, matches_played_since=len([
            x for x in self._served(s, p["team"]) if self.state["fixtures"][x]["status"] == "played"]), **s)
            for sk, s in sorted(self.state["suspensions"].items()) if s["player"] == k]
        row["team_next_fixtures"] = [{"fixture_id": fk, "date": f["date"], "kickoff": f["kickoff"],
                                      "opponent": self._team_name(f["away"] if f["home"] == p["team"] else f["home"])}
                                     for _, _, fk, f in sorted((f["date"], f["kickoff"], fk, f)
                                                               for fk, f in self.state["fixtures"].items()
                                                               if p["team"] in (f["home"], f["away"])
                                                               and f["status"] == "scheduled"
                                                               and f["date"] >= self.state["today"])]
        return row

    def _slot_ok(self, fk, f):
        """Validate date/kickoff/venue/pitch/team clashes for fixture record f (key fk, may be new)."""
        d = _pdate(f["date"])
        if d < self._today():
            raise _Rule("%s is in the past (today is %s)" % (f["date"], self.state["today"]))
        v = self._get("venues", f["venue"], "venue")
        if f["date"] in v["closed_dates"]:
            raise _Rule("%s is closed on %s" % (v["name"], f["date"]))
        if f["pitch"] > v["pitches"]:
            raise _Rule("%s has pitches 1-%d" % (v["name"], v["pitches"]))
        start = _ptime(f["kickoff"], "kickoff")
        if start % 15:
            raise _Rule("kick-off must be on a 15-minute mark")
        end = start + self._div(f)["match_minutes"]
        if start < _ptime(v["open"]) or end > _ptime(v["close"]):
            raise _Rule("%s-%s does not fit %s's opening hours (%s-%s)" % (
                f["kickoff"], _fmt_time(end), v["name"], v["open"], v["close"]))
        if end > _ptime(self.state["policy"]["dusk"]) and f["pitch"] not in v["floodlit"]:
            raise _Rule("the match ends at %s, after dusk (%s), and pitch %d at %s has no floodlights" % (
                _fmt_time(end), self.state["policy"]["dusk"], f["pitch"], v["name"]))
        gap = self.state["policy"]["turnaround"]
        for k, o in sorted(self.state["fixtures"].items()):
            if k == fk or o["status"] != "scheduled" or o["date"] != f["date"]:
                continue
            if o["venue"] == f["venue"] and o["pitch"] == f["pitch"]:
                os, oe = _ptime(o["kickoff"]), self._end(o)
                if start < oe + gap and os < end + gap:
                    raise _Rule("pitch %d at %s is in use %s-%s by %s (turnaround %d min)" % (
                        f["pitch"], v["name"], o["kickoff"], _fmt_time(oe), k, gap))
            for t in (f["home"], f["away"]):
                if t in (o["home"], o["away"]):
                    raise _Rule("%s already plays on %s (%s)" % (self._team_name(t), f["date"], k))

    def _t_schedule_fixture(self, a):
        h, w = self._team(a["home_team"]), self._team(a["away_team"])
        if a["home_team"] == a["away_team"]:
            raise _Rule("a team cannot play itself")
        if h["division"] != w["division"]:
            raise _Rule("%s (%s) and %s (%s) are in different divisions" % (h["name"], h["division"], w["name"],
                                                                            w["division"]))
        f = {"division": h["division"], "home": a["home_team"], "away": a["away_team"], "date": a["date"],
             "kickoff": a["kickoff"], "venue": a["venue_id"], "pitch": a["pitch"], "status": "scheduled",
             "referee": "", "assistant_1": "", "assistant_2": "", "postpone_reason": "", "result": "", "cards": []}
        self._slot_ok("", f)
        k = self._new_id("next_fixture", "FX")
        self.state["fixtures"][k] = f
        return {"ok": True, "fixture": self._fix_row(k)}

    def _t_reschedule_fixture(self, a):
        k = a["fixture_id"]
        f = self._fix(k)
        if f["status"] not in ("scheduled", "postponed"):
            raise _Rule("fixture %s is %s" % (k, f["status"]))
        new = dict(f, date=a["date"], kickoff=a["kickoff"], venue=a.get("venue_id", f["venue"]),
                   pitch=a.get("pitch", f["pitch"]), status="scheduled", postpone_reason="")
        if f["status"] == "scheduled" and all(new[x] == f[x] for x in ("date", "kickoff", "venue", "pitch")):
            raise _Rule("fixture %s is already scheduled then and there" % k)
        self._slot_ok(k, new)
        if (new["date"], new["kickoff"]) != (f["date"], f["kickoff"]):
            new.update(referee="", assistant_1="", assistant_2="")
        f.update(new)
        return {"ok": True, "fixture": self._fix_row(k)}

    def _t_postpone_fixture(self, a):
        k = a["fixture_id"]
        f = self._fix(k)
        if f["status"] != "scheduled":
            raise _Rule("fixture %s is %s" % (k, f["status"]))
        f.update(status="postponed", postpone_reason=a["reason"], referee="", assistant_1="", assistant_2="")
        return {"ok": True, "fixture": self._fix_row(k)}

    def _t_assign_referee(self, a):
        k, rid, role = a["fixture_id"], a["referee_id"], a["role"]
        f = self._fix(k)
        r = self._get("referees", rid, "referee")
        if f["status"] != "scheduled":
            raise _Rule("fixture %s is %s" % (k, f["status"]))
        if f[role] == rid:
            raise _Rule("%s is already the %s of %s" % (rid, role, k))
        for other in _ROLES:
            if other != role and f[other] == rid:
                raise _Rule("%s is already the %s of %s" % (rid, other, k))
        if role == "referee" and r["grade"] < self._div(f)["min_referee_grade"]:
            raise _Rule("a %s referee needs grade %d or higher; %s is grade %d" % (
                f["division"], self._div(f)["min_referee_grade"], rid, r["grade"]))
        if not self._available(r, f["date"]):
            raise _Rule("%s (%s) is not available on %s" % (rid, r["name"], f["date"]))
        for t in (f["home"], f["away"]):
            if r["club"] and self.state["teams"][t]["club"] == r["club"]:
                raise _Rule("%s (%s) belongs to %s, the club of %s" % (
                    rid, r["name"], self.state["clubs"][r["club"]], self._team_name(t)))
        start, end, n = _ptime(f["kickoff"]), self._end(f), 0
        for ok, o in sorted(self.state["fixtures"].items()):
            if ok == k or o["status"] != "scheduled" or o["date"] != f["date"] or rid not in [o[x] for x in _ROLES]:
                continue
            n += 1
            gap = 30 if o["venue"] == f["venue"] else 60
            os, oe = _ptime(o["kickoff"]), self._end(o)
            if start < oe + gap and os < end + gap:
                raise _Rule("%s officiates %s %s-%s at %s; %d minutes are needed between fixtures" % (
                    rid, ok, o["kickoff"], _fmt_time(oe), o["venue"], gap))
        if n >= 2:
            raise _Rule("%s already has 2 fixtures on %s" % (rid, f["date"]))
        replaced = f[role]
        f[role] = rid
        return {"ok": True, "replaced": replaced, "fixture": self._fix_row(k)}

    def _t_record_result(self, a):
        k = a["fixture_id"]
        f = self._fix(k)
        if f["status"] != "scheduled":
            raise _Rule("fixture %s is %s" % (k, f["status"]))
        if f["date"] > self.state["today"]:
            raise _Rule("fixture %s is on %s and has not been played yet" % (k, f["date"]))
        cards = []
        for c in a.get("cards", []):
            p = self._get("players", c["player_id"], "player")
            if p["team"] not in (f["home"], f["away"]):
                raise _Rule("%s plays for %s, not in fixture %s" % (c["player_id"], self._team_name(p["team"]), k))
            sheet = [s for s in self.state["team_sheets"].values() if s["fixture"] == k and s["team"] == p["team"]]
            if sheet and c["player_id"] not in [x["player_id"] for x in sheet[0]["players"]]:
                raise _Rule("%s is not on %s's team sheet for %s" % (c["player_id"], p["team"], k))
            cards.append({"player_id": c["player_id"], "card": c["card"], "minute": c["minute"]})
        f["cards"] = sorted(cards, key=lambda c: (c["minute"], c["player_id"], c["card"]))
        f["result"] = "%d-%d" % (a["home_goals"], a["away_goals"])
        f["status"] = "played"
        return {"ok": True, "fixture": self._fix_row(k)}

    def _t_issue_suspension(self, a):
        k = a["player_id"]
        p = self._get("players", k, "player")
        if p["status"] != "active":
            raise _Rule("%s is %s" % (k, p["status"]))
        pol = self.state["policy"]
        prior = [s for s in self.state["suspensions"].values() if s["player"] == k and s["reason"] == a["reason"]]
        if a["reason"] == "red_card":
            if self._reds(k) <= len(prior):
                raise _Rule("%s has no red card in a recorded result that still needs a suspension" % k)
            if a["matches"] != pol["red_card_matches"]:
                raise _Rule("a red_card suspension is %d match(es)" % pol["red_card_matches"])
        elif a["reason"] == "accumulated_yellows":
            if self._yellows(k) < pol["yellow_limit"] * (len(prior) + 1):
                raise _Rule("%s has %d season yellow cards; the next accumulated_yellows suspension is due at %d" % (
                    k, self._yellows(k), pol["yellow_limit"] * (len(prior) + 1)))
            if a["matches"] != pol["yellow_ban_matches"]:
                raise _Rule("an accumulated_yellows suspension is %d match(es)" % pol["yellow_ban_matches"])
        n = self._new_id("next_suspension", "SU")
        self.state["suspensions"][n] = {"player": k, "matches": a["matches"], "reason": a["reason"],
                                        "issued_on": self.state["today"]}
        return {"ok": True, "suspension_id": n, "suspension": self.state["suspensions"][n]}

    def _age_ok(self, div, dob):
        d = self.state["policy"]["divisions"][div]
        if d["born_on_or_after"] and dob < d["born_on_or_after"]:
            raise _Rule("%s players must be born on or after %s" % (div, d["born_on_or_after"]))
        if d["born_on_or_before"] and dob > d["born_on_or_before"]:
            raise _Rule("%s players must be born on or before %s" % (div, d["born_on_or_before"]))

    def _t_submit_team_sheet(self, a):
        fk, tid = a["fixture_id"], a["team_id"]
        f = self._fix(fk)
        t = self._team(tid)
        if f["status"] != "scheduled":
            raise _Rule("fixture %s is %s" % (fk, f["status"]))
        if tid not in (f["home"], f["away"]):
            raise _Rule("%s does not play in %s" % (tid, fk))
        div = self._div(f)
        ids = [x["player_id"] for x in a["players"]]
        if len(set(ids)) != len(ids):
            raise _Rule("a player is listed twice")
        if len(set(x["shirt"] for x in a["players"])) != len(ids):
            raise _Rule("shirt numbers must be distinct")
        if len(ids) > div["squad_size"]:
            raise _Rule("a %s sheet names at most %d players" % (f["division"], div["squad_size"]))
        starters = [x["player_id"] for x in a["players"] if x.get("starting", True)]
        if len(starters) != div["side_size"]:
            raise _Rule("a %s sheet needs exactly %d starters, got %d" % (f["division"], div["side_size"], len(starters)))
        if a["captain_id"] not in starters:
            raise _Rule("the captain must be one of the starters")
        for pid in ids:
            p = self._get("players", pid, "player")
            if p["team"] != tid or p["status"] != "active":
                raise _Rule("%s is not an active player of %s" % (pid, t["name"]))
            if _plus_days(p["registered_on"], self.state["policy"]["registration_days"]) > f["date"]:
                raise _Rule("%s was registered on %s and is eligible only from %s" % (
                    pid, p["registered_on"], _plus_days(p["registered_on"], self.state["policy"]["registration_days"])))
            self._age_ok(f["division"], p["dob"])
            su = self._suspended_for(pid, fk)
            if su:
                raise _Rule("%s is suspended for %s (%s)" % (pid, fk, su))
        players = sorted([{"player_id": x["player_id"], "shirt": x["shirt"], "starting": x.get("starting", True)}
                          for x in a["players"]], key=lambda x: x["shirt"])
        for sk, s in sorted(self.state["team_sheets"].items()):
            if s["fixture"] == fk and s["team"] == tid:
                s["players"], s["captain"] = players, a["captain_id"]
                return {"ok": True, "sheet_id": sk, "replaced": True}
        n = self._new_id("next_sheet", "TS")
        self.state["team_sheets"][n] = {"fixture": fk, "team": tid, "players": players, "captain": a["captain_id"]}
        return {"ok": True, "sheet_id": n, "replaced": False}

    def _t_register_player(self, a):
        tid = a["team_id"]
        t = self._team(tid)
        name = _text(a["name"], "name", 60)
        _pdate(a["dob"], "dob")
        self._age_ok(t["division"], a["dob"])
        for k, p in sorted(self.state["players"].items()):
            if p["name"].lower() == name.lower() and p["dob"] == a["dob"]:
                raise _Rule("%s born %s is already registered (%s)" % (name, a["dob"], k))
        n = self._new_id("next_player", "PL")
        self.state["players"][n] = {"name": name, "team": tid, "dob": a["dob"], "registered_on": self.state["today"],
                                    "position": a.get("position", ""), "status": "active", "prior_yellows": 0}
        return {"ok": True, "player": self._player_row(n)}


def _league():
    clubs = {"CL-1": "Riverside Rovers", "CL-2": "Hilltop United", "CL-3": "Northfield Athletic", "CL-4": "Ashby Town",
             "CL-5": "Saint Brendan FC"}
    teams = dict((k, {"name": n, "division": d, "club": c}) for k, n, d, c in [
        ("TM-01", "Riverside Rovers", "senior", "CL-1"), ("TM-02", "Hilltop United", "senior", "CL-2"),
        ("TM-03", "Northfield Athletic", "senior", "CL-3"), ("TM-04", "Ashby Town", "senior", "CL-4"),
        ("TM-05", "Ashby Town Reserves", "senior", "CL-4"), ("TM-06", "Saint Brendan FC", "senior", "CL-5"),
        ("TM-07", "Northfield Athletic Reserves", "senior", "CL-3"), ("TM-08", "Hilltop United Reserves", "senior", "CL-2"),
        ("TM-11", "Riverside Rovers U14", "u14", "CL-1"), ("TM-12", "Hilltop United U14", "u14", "CL-2"),
        ("TM-13", "Ashby Town U14", "u14", "CL-4"), ("TM-14", "Saint Brendan U14", "u14", "CL-5"),
        ("TM-21", "Northfield U12", "u12", "CL-3"), ("TM-22", "Saint Brendan U12", "u12", "CL-5")])
    venues = {
        "VN-1": {"name": "Mill Lane Park", "pitches": 3, "floodlit": [1], "open": "09:00", "close": "21:00",
                 "closed_dates": ["2026-11-14"]},
        "VN-2": {"name": "Hilltop Rec", "pitches": 2, "floodlit": [1, 2], "open": "09:00", "close": "22:00",
                 "closed_dates": []},
        "VN-3": {"name": "Northfield Academy", "pitches": 1, "floodlit": [], "open": "09:00", "close": "17:00",
                 "closed_dates": ["2026-12-26"]},
        "VN-4": {"name": "Ashby Sports Hub", "pitches": 4, "floodlit": [3, 4], "open": "08:00", "close": "21:30",
                 "closed_dates": ["2026-11-21"]},
    }
    referees = dict((k, {"name": n, "grade": g, "club": c, "days": d, "unavailable": u}) for k, n, g, c, d, u in [
        ("RF-01", "Dan Hughes", 4, "CL-1", ["Sat", "Sun"], []),
        ("RF-02", "Priya Nandi", 3, "", ["Wed", "Sat", "Sun"], []),
        ("RF-03", "Tom Kerr", 2, "CL-3", ["Sat"], []),
        ("RF-04", "Leah Moss", 3, "CL-2", ["Sat", "Sun"], []),
        ("RF-05", "Ola Ade", 1, "", ["Sat", "Sun"], []),
        ("RF-06", "Ben Cole", 5, "", ["Sun"], ["2026-11-15"]),
        ("RF-07", "Sara Lund", 2, "CL-5", ["Sat", "Sun"], []),
        ("RF-08", "Kai Brandt", 2, "CL-1", ["Wed", "Sat"], []),
        ("RF-09", "Nina Ross", 2, "", ["Wed"], [])])
    players = {}
    for k, name, team, dob, reg, pos, prior in [
        ("PL-104", "Marcus Webb", "TM-01", "1998-04-12", "2025-08-01", "midfielder", 4),
        ("PL-108", "Jamie Fox", "TM-01", "2001-09-30", "2025-08-01", "defender", 2),
        ("PL-110", "Ollie Grant", "TM-01", "1996-01-05", "2024-07-20", "defender", 5),
        ("PL-111", "Kieran Lowe", "TM-01", "2003-03-03", "2026-08-10", "forward", 0),
        ("PL-210", "Rhys Barker", "TM-02", "1999-02-17", "2025-08-05", "midfielder", 1),
        ("PL-212", "Callum Doyle", "TM-02", "2000-12-01", "2025-08-05", "forward", 3),
        ("PL-301", "Dev Patel", "TM-03", "1997-07-07", "2023-08-01", "midfielder", 4),
        ("PL-305", "Sam Hart", "TM-03", "2002-05-19", "2025-09-01", "goalkeeper", 0),
        ("PL-401", "Luca Moretti", "TM-04", "1995-10-10", "2022-08-01", "forward", 1),
        ("PL-501", "Ava Price", "TM-21", "2015-02-11", "2026-08-20", "goalkeeper", 0),
        ("PL-502", "Noah Kim", "TM-21", "2015-05-02", "2026-08-20", "defender", 0),
        ("PL-503", "Leo Grant", "TM-21", "2014-11-23", "2026-08-20", "midfielder", 0),
        ("PL-504", "Mia Torres", "TM-21", "2015-01-30", "2026-08-20", "defender", 0),
        ("PL-505", "Isla Webb", "TM-21", "2015-07-14", "2026-08-20", "midfielder", 0),
        ("PL-506", "Ethan Ross", "TM-21", "2014-10-02", "2026-08-20", "forward", 0),
        ("PL-507", "Zara Khan", "TM-21", "2015-03-09", "2026-09-01", "forward", 0),
        ("PL-508", "Finn Doyle", "TM-21", "2015-08-21", "2026-09-01", "defender", 0),
        ("PL-509", "Ruby Shaw", "TM-21", "2015-04-04", "2026-11-10", "midfielder", 0),
        ("PL-601", "Sean Byrne", "TM-22", "2014-12-12", "2026-08-15", "midfielder", 0),
        ("PL-602", "Aoife Kelly", "TM-22", "2015-02-02", "2026-08-15", "goalkeeper", 0),
        ("PL-603", "Tom Walsh", "TM-22", "2015-06-06", "2026-08-15", "defender", 0),
        ("PL-604", "Cian Ryan", "TM-22", "2014-09-09", "2026-08-15", "defender", 0),
        ("PL-605", "Niamh Burke", "TM-22", "2015-03-03", "2026-08-15", "midfielder", 0),
        ("PL-606", "Jack Daly", "TM-22", "2015-01-01", "2026-08-15", "forward", 0),
        ("PL-607", "Ella Nolan", "TM-22", "2015-05-05", "2026-08-15", "forward", 0),
        ("PL-608", "Liam Quinn", "TM-22", "2015-07-07", "2026-08-15", "defender", 0),
        ("PL-609", "Orla Hayes", "TM-22", "2015-08-08", "2026-09-10", "midfielder", 0),
        ("PL-610", "Darragh Lynch", "TM-22", "2015-04-14", "2026-11-09", "forward", 0),
        ("PL-701", "Aaron Clarke", "TM-14", "2013-02-02", "2025-08-15", "forward", 0),
        ("PL-702", "Ben Shaw", "TM-14", "2012-10-10", "2025-08-15", "defender", 0),
    ]:
        players[k] = {"name": name, "team": team, "dob": dob, "registered_on": reg, "position": pos,
                      "status": "active", "prior_yellows": prior}
    players["PL-111"]["status"] = "left"
    fixtures = {}
    for k, div, home, away, date, ko, venue, pitch, status, ref, a1, a2, result in [
        ("FX-101", "senior", "TM-01", "TM-04", "2026-10-25", "11:00", "VN-1", 1, "played", "RF-02", "", "", "2-0"),
        ("FX-102", "u12", "TM-22", "TM-21", "2026-10-25", "10:00", "VN-3", 1, "played", "RF-05", "", "", "3-3"),
        ("FX-201", "senior", "TM-01", "TM-02", "2026-11-08", "11:00", "VN-2", 1, "scheduled", "RF-06", "", "", ""),
        ("FX-203", "u12", "TM-22", "TM-21", "2026-11-08", "10:00", "VN-3", 1, "scheduled", "RF-05", "", "", ""),
        ("FX-204", "senior", "TM-03", "TM-01", "2026-11-07", "14:00", "VN-3", 1, "scheduled", "RF-04", "", "", ""),
        ("FX-230", "u14", "TM-12", "TM-11", "2026-11-07", "10:00", "VN-2", 2, "postponed", "", "", "", ""),
        ("FX-205", "senior", "TM-03", "TM-04", "2026-11-14", "14:00", "VN-3", 1, "scheduled", "RF-04", "", "", ""),
        ("FX-210", "senior", "TM-01", "TM-02", "2026-11-14", "13:00", "VN-1", 2, "scheduled", "RF-02", "", "", ""),
        ("FX-211", "u14", "TM-11", "TM-12", "2026-11-14", "10:00", "VN-1", 2, "scheduled", "RF-07", "", "", ""),
        ("FX-212", "senior", "TM-05", "TM-06", "2026-11-14", "16:00", "VN-1", 1, "scheduled", "", "", "", ""),
        ("FX-213", "u14", "TM-13", "TM-14", "2026-11-14", "10:00", "VN-4", 1, "scheduled", "RF-05", "", "", ""),
        ("FX-216", "senior", "TM-07", "TM-08", "2026-11-14", "16:15", "VN-4", 3, "scheduled", "RF-01", "", "", ""),
        ("FX-220", "u12", "TM-21", "TM-22", "2026-11-15", "10:00", "VN-3", 1, "scheduled", "", "", "", ""),
        ("FX-221", "senior", "TM-01", "TM-02", "2026-11-15", "11:00", "VN-2", 1, "scheduled", "", "", "", ""),
        ("FX-222", "u14", "TM-12", "TM-14", "2026-11-15", "13:00", "VN-2", 2, "scheduled", "", "", "", ""),
        ("FX-223", "senior", "TM-03", "TM-04", "2026-11-15", "14:30", "VN-1", 1, "scheduled", "RF-04", "", "", ""),
        ("FX-214", "u14", "TM-13", "TM-14", "2026-11-21", "10:00", "VN-4", 2, "scheduled", "RF-05", "", "", ""),
        ("FX-206", "u14", "TM-11", "TM-12", "2026-11-21", "12:00", "VN-2", 1, "scheduled", "RF-07", "", "", ""),
        ("FX-225", "u12", "TM-21", "TM-22", "2026-11-21", "10:00", "VN-3", 1, "scheduled", "", "", "", ""),
        ("FX-231", "senior", "TM-05", "TM-06", "2026-11-25", "18:00", "VN-2", 1, "scheduled", "RF-02", "", "", ""),
        ("FX-232", "u14", "TM-13", "TM-14", "2026-11-25", "18:30", "VN-2", 2, "scheduled", "", "", "", ""),
        ("FX-240", "senior", "TM-02", "TM-04", "2026-11-28", "12:00", "VN-2", 1, "scheduled", "RF-02", "", "", ""),
        ("FX-241", "u14", "TM-12", "TM-14", "2026-11-28", "12:30", "VN-2", 2, "scheduled", "", "", "", ""),
    ]:
        fixtures[k] = {"division": div, "home": home, "away": away, "date": date, "kickoff": ko, "venue": venue,
                       "pitch": pitch, "status": status, "referee": ref, "assistant_1": a1, "assistant_2": a2,
                       "postpone_reason": "pitch_unfit" if status == "postponed" else "", "result": result,
                       "cards": []}
    fixtures["FX-101"]["cards"] = [{"player_id": "PL-110", "card": "yellow", "minute": 40}]
    sheets = {
        "TS-1": {"fixture": "FX-220", "team": "TM-22", "captain": "PL-601", "players": [
            {"player_id": p, "shirt": n, "starting": n <= 7} for p, n in [
                ("PL-601", 1), ("PL-602", 2), ("PL-603", 3), ("PL-604", 4), ("PL-605", 5), ("PL-606", 6),
                ("PL-607", 7), ("PL-608", 12), ("PL-609", 14)]]},
        "TS-2": {"fixture": "FX-225", "team": "TM-21", "captain": "PL-503", "players": [
            {"player_id": p, "shirt": n, "starting": n <= 7} for p, n in [
                ("PL-501", 1), ("PL-502", 2), ("PL-503", 3), ("PL-504", 4), ("PL-505", 5), ("PL-506", 6),
                ("PL-507", 7), ("PL-508", 12)]]},
    }
    st = {"today": "2026-11-12",
          "policy": {"divisions": {
              "senior": {"side_size": 11, "squad_size": 16, "match_minutes": 105, "min_referee_grade": 3,
                         "born_on_or_after": "", "born_on_or_before": "2010-08-31"},
              "u14": {"side_size": 9, "squad_size": 14, "match_minutes": 80, "min_referee_grade": 2,
                      "born_on_or_after": "2012-09-01", "born_on_or_before": ""},
              "u12": {"side_size": 7, "squad_size": 10, "match_minutes": 60, "min_referee_grade": 1,
                      "born_on_or_after": "2014-09-01", "born_on_or_before": ""}},
              "turnaround": 15, "dusk": "16:30", "registration_days": 7, "red_card_matches": 1,
              "yellow_limit": 5, "yellow_ban_matches": 1},
          "clubs": clubs, "teams": teams, "venues": venues, "referees": referees, "players": players,
          "fixtures": fixtures, "team_sheets": sheets,
          "suspensions": {"SU-1": {"player": "PL-110", "matches": 1, "reason": "accumulated_yellows",
                                   "issued_on": "2026-10-20"}},
          "next_fixture": 300, "next_player": 900, "next_sheet": 10, "next_suspension": 10}
    return st


def _sheet(fixture, team, captain, entries):
    return {"fixture": fixture, "team": team, "captain": captain,
            "players": sorted([{"player_id": p, "shirt": n, "starting": s} for p, n, s in entries],
                              key=lambda x: x["shirt"])}


def _fx_new(div, home, away, date, ko, venue, pitch, ref=""):
    return {"division": div, "home": home, "away": away, "date": date, "kickoff": ko, "venue": venue, "pitch": pitch,
            "status": "scheduled", "referee": ref, "assistant_1": "", "assistant_2": "", "postpone_reason": "",
            "result": "", "cards": []}


_GT = "2026-11-12"   # league "today"
_TS2 = [("PL-501", 1, True), ("PL-502", 2, True), ("PL-503", 3, True), ("PL-504", 4, True), ("PL-505", 5, True),
        ("PL-506", 6, True), ("PL-507", 7, True), ("PL-508", 12, False)]

# ---- single-call S19-S24
_add("S19", "league", _league(),
     "Result from last Sunday, fixture FX-201 (Riverside Rovers v Hilltop United, attendance 212): Riverside won "
     "2-1. Cards: Marcus Webb (PL-104) yellow in the 34th minute; Rhys Barker (PL-210) yellow on 61 minutes; "
     "Callum Doyle (PL-212) straight red in stoppage time at 90+3. Please record the result with all three cards.",
     [["record_result", {"fixture_id": "FX-201", "home_goals": 2, "away_goals": 1, "cards": [
         {"player_id": "PL-104", "card": "yellow", "minute": 34}, {"player_id": "PL-210", "card": "yellow", "minute": 61},
         {"player_id": "PL-212", "card": "red", "minute": 93}]}]],
     [["record_result", {"fixture_id": "FX-201", "home_goals": 2, "away_goals": 1, "cards": [
         {"player_id": "PL-104", "card": "yellow", "minute": 34}, {"player_id": "PL-210", "card": "yellow", "minute": 61},
         {"player_id": "PL-212", "card": "red", "minute": 90}]}]],
     lambda s, i: _final(s, i, changes={"fixtures": {"FX-201": {"status": "played", "result": "2-1", "cards": [
         {"player_id": "PL-104", "card": "yellow", "minute": 34}, {"player_id": "PL-210", "card": "yellow", "minute": 61},
         {"player_id": "PL-212", "card": "red", "minute": 93}]}}}),
     "OUX", mentions=["Riverside Rovers", "Hilltop United", "Marcus Webb", "Rhys Barker", "Callum Doyle"])

_add("S20", "league", _league(),
     "Ashby Sports Hub (VN-4) is shut on Saturday 21 November, so fixture FX-214 (Ashby Town U14 v Saint Brendan "
     "U14, 10:00, pitch 2) has to move to the following day. Same venue and same pitch, kicking off at half past "
     "ten in the morning.",
     [["reschedule_fixture", {"fixture_id": "FX-214", "date": "2026-11-22", "kickoff": "10:30"}]],
     [["postpone_fixture", {"fixture_id": "FX-214", "reason": "pitch_unfit"}],
      ["reschedule_fixture", {"fixture_id": "FX-214", "date": "2026-11-22", "kickoff": "10:00"}]],
     lambda s, i: _final(s, i, changes={"fixtures": {"FX-214": {"date": "2026-11-22", "kickoff": "10:30",
                                                                "referee": ""}}}),
     "TDU", mentions=["Ashby Sports Hub", "Ashby Town U14", "Saint Brendan U14"])

_add("S21", "league", _league(),
     "The council has shut the pitch at Northfield Academy for fixture FX-205 (Northfield Athletic v Ashby Town, "
     "Saturday 14:00) because its drainage has collapsed - the forecast is dry and both teams and the referee are "
     "available. Call the match off for now with the reason that fits; a new date will be set later.",
     [["postpone_fixture", {"fixture_id": "FX-205", "reason": "pitch_unfit"}]],
     [["postpone_fixture", {"fixture_id": "FX-205", "reason": "bad_weather"}]],
     lambda s, i: _final(s, i, changes={"fixtures": {"FX-205": {"status": "postponed", "postpone_reason": "pitch_unfit",
                                                                "referee": ""}}}),
     "NT", mentions=["Northfield Academy", "Northfield Athletic", "Ashby Town"])

_add("S22", "league", _league(),
     "For fixture FX-206 (Riverside Rovers U14 v Hilltop United U14 on 21 November, referee Sara Lund), Priya "
     "Nandi (RF-02) will run the line on the far side as the second assistant referee. Please put her on the "
     "fixture. Nobody is the first assistant yet.",
     [["assign_referee", {"fixture_id": "FX-206", "referee_id": "RF-02", "role": "assistant_2"}]],
     [["assign_referee", {"fixture_id": "FX-206", "referee_id": "RF-02", "role": "assistant_1"}]],
     lambda s, i: _final(s, i, changes={"fixtures": {"FX-206": {"assistant_2": "RF-02"}}}),
     "NT", mentions=["Sara Lund", "Priya Nandi"])

_add("S23", "league", _league(),
     "Team sheet for Northfield U12 (TM-21) for Sunday's fixture FX-220. Starting: Ava Price PL-501 in goal with "
     "shirt 1, Noah Kim PL-502 shirt 2, Leo Grant PL-503 shirt 3 (captain), Mia Torres PL-504 shirt 4, Isla Webb "
     "PL-505 shirt 8, Ethan Ross PL-506 shirt 9, Zara Khan PL-507 shirt 10. On the bench: Finn Doyle PL-508 with "
     "shirt 12. Ruby Shaw only joined this week, so she is not named.",
     [["submit_team_sheet", {"fixture_id": "FX-220", "team_id": "TM-21", "captain_id": "PL-503", "players": [
         {"player_id": "PL-501", "shirt": 1}, {"player_id": "PL-502", "shirt": 2}, {"player_id": "PL-503", "shirt": 3},
         {"player_id": "PL-504", "shirt": 4}, {"player_id": "PL-505", "shirt": 8}, {"player_id": "PL-506", "shirt": 9},
         {"player_id": "PL-507", "shirt": 10}, {"player_id": "PL-508", "shirt": 12, "starting": False}]}]],
     [["submit_team_sheet", {"fixture_id": "FX-220", "team_id": "TM-21", "captain_id": "PL-503", "players": [
         {"player_id": "PL-501", "shirt": 1}, {"player_id": "PL-502", "shirt": 2}, {"player_id": "PL-503", "shirt": 3},
         {"player_id": "PL-504", "shirt": 4}, {"player_id": "PL-505", "shirt": 5}, {"player_id": "PL-506", "shirt": 6},
         {"player_id": "PL-507", "shirt": 7}, {"player_id": "PL-508", "shirt": 12, "starting": False}]}]],
     lambda s, i: _final(s, i, new={"team_sheets": [_sheet("FX-220", "TM-21", "PL-503", [
         ("PL-501", 1, True), ("PL-502", 2, True), ("PL-503", 3, True), ("PL-504", 4, True), ("PL-505", 8, True),
         ("PL-506", 9, True), ("PL-507", 10, True), ("PL-508", 12, False)])]}),
     "OXP", mentions=["Northfield U12", "Ava Price", "Finn Doyle", "Ruby Shaw"])

_add("S24", "league", _league(),
     "Please register a new signing for Northfield U12 (TM-21): Maya Chen, born on the 3rd of March 2015. She "
     "plays between the posts, so record her usual position too.",
     [["register_player", {"name": "Maya Chen", "team_id": "TM-21", "dob": "2015-03-03", "position": "goalkeeper"}]],
     [["register_player", {"name": "Maya Chen", "team_id": "TM-21", "dob": "2015-03-03"}]],
     lambda s, i: _final(s, i, new={"players": [{"name": "Maya Chen", "team": "TM-21", "dob": "2015-03-03",
                                                 "registered_on": _GT, "position": "goalkeeper", "status": "active",
                                                 "prior_yellows": 0}]}),
     "DNP", mentions=["Northfield U12"], creates=["Maya Chen"])

# ---- multi-step M19-M24
_add("M19", "league", _league(),
     "Mill Lane Park has been closed for this Saturday, 14 November. Move every fixture that was due to be played "
     "there that day to Ashby Sports Hub, same date and same kick-off, each onto the lowest-numbered pitch there "
     "that the rules allow. Officials stay as they are. If a fixture can't be fitted in at all, postpone it as "
     "pitch unfit. Nothing else should change.",
     [["get_league_rules", {}], ["list_venues", {}], ["list_fixtures", {"venue_id": "VN-1", "date": "2026-11-14"}],
      ["list_fixtures", {"venue_id": "VN-4", "date": "2026-11-14"}],
      ["reschedule_fixture", {"fixture_id": "FX-211", "date": "2026-11-14", "kickoff": "10:00", "venue_id": "VN-4",
                              "pitch": 1}],
      ["reschedule_fixture", {"fixture_id": "FX-211", "date": "2026-11-14", "kickoff": "10:00", "venue_id": "VN-4",
                              "pitch": 2}],
      ["reschedule_fixture", {"fixture_id": "FX-210", "date": "2026-11-14", "kickoff": "13:00", "venue_id": "VN-4",
                              "pitch": 1}],
      ["reschedule_fixture", {"fixture_id": "FX-212", "date": "2026-11-14", "kickoff": "16:00", "venue_id": "VN-4",
                              "pitch": 3}],
      ["reschedule_fixture", {"fixture_id": "FX-212", "date": "2026-11-14", "kickoff": "16:00", "venue_id": "VN-4",
                              "pitch": 4}],
      ["list_fixtures", {"venue_id": "VN-4", "date": "2026-11-14"}]],
     [["reschedule_fixture", {"fixture_id": "FX-211", "date": "2026-11-14", "kickoff": "10:00", "venue_id": "VN-4",
                              "pitch": 2}],
      ["reschedule_fixture", {"fixture_id": "FX-210", "date": "2026-11-14", "kickoff": "13:00", "venue_id": "VN-4",
                              "pitch": 1}],
      ["postpone_fixture", {"fixture_id": "FX-212", "reason": "pitch_unfit"}]],
     lambda s, i: _final(s, i, changes={"fixtures": {
         "FX-211": {"venue": "VN-4", "pitch": 2}, "FX-210": {"venue": "VN-4", "pitch": 1},
         "FX-212": {"venue": "VN-4", "pitch": 4}}}),
     "abdf", ref_errors=[4, 7], mentions=["Mill Lane Park", "Ashby Sports Hub"])

_add("M20", "league", _league(),
     "Several fixtures this Sunday, 15 November, still have no referee. Going through them in kick-off order, "
     "appoint to each one the highest-graded referee that the league rules allow at that point (ties: lowest "
     "referee id). Fixtures that already have a referee keep theirs, and no assistants are needed this weekend. "
     "Nothing else should change.",
     [["get_league_rules", {}], ["list_fixtures", {"date": "2026-11-15"}],
      ["list_referees", {"available_on": "2026-11-15"}],
      ["assign_referee", {"fixture_id": "FX-220", "referee_id": "RF-01", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-221", "referee_id": "RF-01", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-221", "referee_id": "RF-02", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-222", "referee_id": "RF-01", "role": "referee"}],
      ["list_fixtures", {"date": "2026-11-15"}]],
     [["assign_referee", {"fixture_id": "FX-220", "referee_id": "RF-06", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-220", "referee_id": "RF-02", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-221", "referee_id": "RF-04", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-222", "referee_id": "RF-01", "role": "referee"}]],
     lambda s, i: _final(s, i, changes={"fixtures": {
         "FX-220": {"referee": "RF-01"}, "FX-221": {"referee": "RF-02"}, "FX-222": {"referee": "RF-01"}}}),
     "abdf", ref_errors=[4])

_add("M21", "league", _league(),
     "Last Sunday's under-12 match FX-203 finished Saint Brendan U12 1, Northfield U12 4, and Cian Ryan was sent "
     "off in the 50th minute. Record the result with that card and give him the suspension the rules require. "
     "If that keeps him out of a team sheet his club has already submitted, replace him in that starting line-up "
     "with the substitute wearing the lowest shirt number (who keeps that number) and leave the rest of the sheet "
     "as it is. Nothing else should change.",
     [["get_league_rules", {}], ["find_players", {"query": "Cian Ryan"}],
      ["record_result", {"fixture_id": "FX-203", "home_goals": 1, "away_goals": 4,
                         "cards": [{"player_id": "PL-604", "card": "red", "minute": 50}]}],
      ["issue_suspension", {"player_id": "PL-604", "matches": 1, "reason": "red_card"}],
      ["get_player", {"player_id": "PL-604"}], ["list_fixtures", {"team_id": "TM-22", "status": "scheduled"}],
      ["submit_team_sheet", {"fixture_id": "FX-220", "team_id": "TM-22", "captain_id": "PL-601", "players": [
          {"player_id": "PL-601", "shirt": 1}, {"player_id": "PL-602", "shirt": 2}, {"player_id": "PL-603", "shirt": 3},
          {"player_id": "PL-605", "shirt": 5}, {"player_id": "PL-606", "shirt": 6}, {"player_id": "PL-607", "shirt": 7},
          {"player_id": "PL-608", "shirt": 12}, {"player_id": "PL-609", "shirt": 14, "starting": False}]}]],
     [["record_result", {"fixture_id": "FX-203", "home_goals": 1, "away_goals": 4,
                         "cards": [{"player_id": "PL-604", "card": "red", "minute": 50}]}],
      ["issue_suspension", {"player_id": "PL-604", "matches": 3, "reason": "misconduct"}]],
     lambda s, i: _final(s, i, changes={
         "fixtures": {"FX-203": {"status": "played", "result": "1-4",
                                 "cards": [{"player_id": "PL-604", "card": "red", "minute": 50}]}},
         "team_sheets": {"TS-1": {"players": _sheet("FX-220", "TM-22", "PL-601", [
             ("PL-601", 1, True), ("PL-602", 2, True), ("PL-603", 3, True), ("PL-605", 5, True), ("PL-606", 6, True),
             ("PL-607", 7, True), ("PL-608", 12, True), ("PL-609", 14, False)])["players"]}}},
         new={"suspensions": [{"player": "PL-604", "matches": 1, "reason": "red_card", "issued_on": _GT}]}),
     "bcef", mentions=["Saint Brendan U12", "Northfield U12", "Cian Ryan"], seq=[[2, 3]])

_add("M22", "league", _league(),
     "Northfield U12 have a new player, Jack Byrne, born 10 June 2015, who usually plays up front. Register him, "
     "then add him as a substitute wearing shirt 15 to the team sheet the club has already submitted for the "
     "first upcoming fixture he is eligible to play in. Everyone already on that sheet stays exactly as they are. "
     "Nothing else should change.",
     [["get_league_rules", {}], ["find_players", {"team_id": "TM-21"}],
      ["register_player", {"name": "Jack Byrne", "team_id": "TM-21", "dob": "2015-06-10", "position": "forward"}],
      ["get_player", {"player_id": "PL-900"}], ["list_fixtures", {"team_id": "TM-21", "status": "scheduled"}],
      ["submit_team_sheet", {"fixture_id": "FX-220", "team_id": "TM-21", "captain_id": "PL-503", "players": [
          {"player_id": "PL-501", "shirt": 1}, {"player_id": "PL-502", "shirt": 2}, {"player_id": "PL-503", "shirt": 3},
          {"player_id": "PL-504", "shirt": 4}, {"player_id": "PL-505", "shirt": 5}, {"player_id": "PL-506", "shirt": 6},
          {"player_id": "PL-507", "shirt": 7}, {"player_id": "PL-900", "shirt": 15, "starting": False}]}],
      ["submit_team_sheet", {"fixture_id": "FX-225", "team_id": "TM-21", "captain_id": "PL-503", "players": [
          {"player_id": "PL-501", "shirt": 1}, {"player_id": "PL-502", "shirt": 2}, {"player_id": "PL-503", "shirt": 3},
          {"player_id": "PL-504", "shirt": 4}, {"player_id": "PL-505", "shirt": 5}, {"player_id": "PL-506", "shirt": 6},
          {"player_id": "PL-507", "shirt": 7}, {"player_id": "PL-508", "shirt": 12, "starting": False},
          {"player_id": "PL-900", "shirt": 15, "starting": False}]}],
      ["list_fixtures", {"team_id": "TM-21", "status": "scheduled"}]],
     [["register_player", {"name": "Jack Byrne", "team_id": "TM-21", "dob": "2015-06-10", "position": "forward"}],
      ["submit_team_sheet", {"fixture_id": "FX-225", "team_id": "TM-21", "captain_id": "PL-503", "players": [
          {"player_id": "PL-501", "shirt": 1}, {"player_id": "PL-502", "shirt": 2}, {"player_id": "PL-503", "shirt": 3},
          {"player_id": "PL-504", "shirt": 4}, {"player_id": "PL-505", "shirt": 5}, {"player_id": "PL-506", "shirt": 6},
          {"player_id": "PL-507", "shirt": 7}, {"player_id": "PL-900", "shirt": 15, "starting": False}]}]],
     lambda s, i: _final(s, i, changes={"team_sheets": {"TS-2": {"players": _sheet(
         "FX-225", "TM-21", "PL-503", _TS2 + [("PL-900", 15, False)])["players"]}}},
         new={"players": [{"name": "Jack Byrne", "team": "TM-21", "dob": "2015-06-10", "registered_on": _GT,
                           "position": "forward", "status": "active", "prior_yellows": 0}]}),
     "bcef", ref_errors=[5], mentions=["Northfield U12"], creates=["Jack Byrne"])

_add("M23", "league", _league(),
     "The postponed under-14 match FX-230 (Hilltop United U14 v Riverside Rovers U14) is to be replayed at Hilltop "
     "Rec on the evening of Wednesday 25 November: use the earliest kick-off from 18:00 that the rules allow on "
     "either pitch there (lower-numbered pitch if both work at that time). Then appoint as referee the "
     "highest-graded referee the rules allow (ties: lowest id). Nothing else should change.",
     [["get_league_rules", {}], ["list_fixtures", {"venue_id": "VN-2", "date": "2026-11-25"}],
      ["reschedule_fixture", {"fixture_id": "FX-230", "date": "2026-11-25", "kickoff": "18:00", "venue_id": "VN-2",
                              "pitch": 1}],
      ["reschedule_fixture", {"fixture_id": "FX-230", "date": "2026-11-25", "kickoff": "20:00", "venue_id": "VN-2",
                              "pitch": 1}],
      ["list_referees", {"available_on": "2026-11-25"}],
      ["assign_referee", {"fixture_id": "FX-230", "referee_id": "RF-02", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-230", "referee_id": "RF-09", "role": "referee"}],
      ["list_fixtures", {"venue_id": "VN-2", "date": "2026-11-25"}]],
     [["reschedule_fixture", {"fixture_id": "FX-230", "date": "2026-11-25", "kickoff": "20:15", "venue_id": "VN-2",
                              "pitch": 2}],
      ["assign_referee", {"fixture_id": "FX-230", "referee_id": "RF-08", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-230", "referee_id": "RF-09", "role": "referee"}]],
     lambda s, i: _final(s, i, changes={"fixtures": {"FX-230": {
         "date": "2026-11-25", "kickoff": "20:00", "venue": "VN-2", "pitch": 1, "status": "scheduled",
         "postpone_reason": "", "referee": "RF-09"}}}),
     "abcef", ref_errors=[2, 5], mentions=["Hilltop United U14", "Riverside Rovers U14", "Hilltop Rec"],
     seq=[[3, 6]])

_add("M24", "league", _league(),
     "Enter the result of last Saturday's FX-204: Northfield Athletic 1, Riverside Rovers 1. Yellow cards: Dev "
     "Patel (23rd minute), Marcus Webb (55th) and Jamie Fox (80th). Then issue every suspension the league rules "
     "now require because of these cards - and only those; earlier cards that were already dealt with need "
     "nothing more. Nothing else should change.",
     [["get_league_rules", {}], ["find_players", {"team_id": "TM-03"}], ["find_players", {"team_id": "TM-01"}],
      ["record_result", {"fixture_id": "FX-204", "home_goals": 1, "away_goals": 1, "cards": [
          {"player_id": "PL-301", "card": "yellow", "minute": 23}, {"player_id": "PL-104", "card": "yellow", "minute": 55},
          {"player_id": "PL-108", "card": "yellow", "minute": 80}]}],
      ["find_players", {"team_id": "TM-01"}],
      ["issue_suspension", {"player_id": "PL-104", "matches": 1, "reason": "accumulated_yellows"}],
      ["issue_suspension", {"player_id": "PL-301", "matches": 1, "reason": "accumulated_yellows"}],
      ["get_player", {"player_id": "PL-301"}]],
     [["record_result", {"fixture_id": "FX-204", "home_goals": 1, "away_goals": 1, "cards": [
         {"player_id": "PL-301", "card": "yellow", "minute": 23}, {"player_id": "PL-104", "card": "yellow", "minute": 55},
         {"player_id": "PL-108", "card": "yellow", "minute": 80}]}],
      ["issue_suspension", {"player_id": "PL-104", "matches": 1, "reason": "accumulated_yellows"}],
      ["issue_suspension", {"player_id": "PL-110", "matches": 1, "reason": "misconduct"}]],
     lambda s, i: _final(s, i, changes={"fixtures": {"FX-204": {"status": "played", "result": "1-1", "cards": [
         {"player_id": "PL-301", "card": "yellow", "minute": 23}, {"player_id": "PL-104", "card": "yellow", "minute": 55},
         {"player_id": "PL-108", "card": "yellow", "minute": 80}]}}},
         new={"suspensions": [{"player": "PL-104", "matches": 1, "reason": "accumulated_yellows", "issued_on": _GT},
                              {"player": "PL-301", "matches": 1, "reason": "accumulated_yellows", "issued_on": _GT}]}),
     "acdf", mentions=["Northfield Athletic", "Riverside Rovers", "Dev Patel", "Marcus Webb", "Jamie Fox"],
     seq=[[3, 5], [3, 6]])

# ---- calibration C4 (multi)
_add("C4", "league", _league(),
     "Riverside Rovers and Northfield Athletic want a senior friendly at Hilltop Rec on Saturday 28 November, "
     "Riverside at home. Schedule it at the earliest kick-off from 12:00 that the rules allow on either pitch "
     "(lower-numbered pitch if both work at that time), then appoint as referee the highest-graded referee the "
     "rules allow (ties: lowest id). Nothing else should change.",
     [["get_league_rules", {}], ["list_venues", {}], ["list_fixtures", {"date": "2026-11-28"}],
      ["schedule_fixture", {"home_team": "TM-01", "away_team": "TM-03", "date": "2026-11-28", "kickoff": "12:00",
                            "venue_id": "VN-2", "pitch": 1}],
      ["schedule_fixture", {"home_team": "TM-01", "away_team": "TM-03", "date": "2026-11-28", "kickoff": "14:00",
                            "venue_id": "VN-2", "pitch": 1}],
      ["list_referees", {"available_on": "2026-11-28"}],
      ["assign_referee", {"fixture_id": "FX-300", "referee_id": "RF-01", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-300", "referee_id": "RF-02", "role": "referee"}],
      ["assign_referee", {"fixture_id": "FX-300", "referee_id": "RF-04", "role": "referee"}],
      ["list_fixtures", {"date": "2026-11-28"}]],
     [["schedule_fixture", {"home_team": "TM-01", "away_team": "TM-03", "date": "2026-11-28", "kickoff": "14:15",
                            "venue_id": "VN-2", "pitch": 2}],
      ["assign_referee", {"fixture_id": "FX-300", "referee_id": "RF-04", "role": "referee"}]],
     lambda s, i: _final(s, i, new={"fixtures": [
         _fx_new("senior", "TM-01", "TM-03", "2026-11-28", "14:00", "VN-2", 1, ref="RF-04")]}),
     "abcf", ref_errors=[3, 6, 7], mentions=["Riverside Rovers", "Northfield Athletic", "Hilltop Rec"])


# --------------------------------------------------------------------------
# 5. museum collections: objects, galleries and light exposure, condition reports, conservation, loans
# --------------------------------------------------------------------------
_OBJ_TYPES = ["painting", "works_on_paper", "photograph", "textile", "furniture", "sculpture", "ceramic", "metalwork"]
_OBJ_STATUS = ["on_display", "in_store", "in_conservation", "on_loan"]
_CONDITIONS = ["good", "fair", "poor"]
_ISSUES = ["crack", "flaking", "fading", "tear", "stain", "corrosion", "insect_damage", "loose_joint"]
_SEVERITY = ["minor", "moderate", "severe"]
_TREATMENTS = ["cleaning", "consolidation", "repair", "reframing", "pest_treatment"]
_LOAN_STATUS = ["draft", "approved", "active", "returned", "cancelled"]
_LOAN_CANCEL = ["partner_withdrew", "condition_concern", "schedule_conflict"]


class MuseumEnv(_Env):
    TOOLS = [
        _fn("get_collections_policy", "Today's date, light limits and annual light-exposure budgets per sensitivity, "
            "which store each kind of object belongs in, the staff, and the collection rules that the write tools "
            "enforce."),
        _fn("search_objects", "Collection objects, filtered, by object id; 10 per page. Shows type, sensitivity, "
            "location, status, display end date, condition, last condition report and light exposure used this year.",
            {"query": _s("Text that must appear in the title or artist."), "type": _s("Only this type.", _OBJ_TYPES),
             "location": _s("Only objects at this gallery or store id, e.g. GL-2 or ST-1."),
             "status": _s("Only this status.", _OBJ_STATUS), "page": _PAGE}),
        _fn("get_object", "One object with its condition reports, loans and conservation treatments.",
            {"object_id": _s("Object id, e.g. OB-1004.")}, ["object_id"]),
        _fn("list_galleries", "Galleries: light level (lux), display hours per day, capacity, open/closed, and the "
            "objects on display in each."),
        _fn("list_loans", "Outgoing loans with partner, dates, objects, courier and light conditions.",
            {"status": _s("Only loans in this status.", _LOAN_STATUS)}),
        _fn("move_object", "Move an object that is in store or on display to a gallery (with a display end date) "
            "or to a store.",
            {"object_id": _s("Object id."), "to_location": _s("Gallery id (GL-...) or store id (ST-...)."),
             "display_until": _s("Last display day, YYYY-MM-DD (required for galleries, not used for stores).")},
            ["object_id", "to_location"]),
        _fn("set_gallery_lux", "Change a gallery's light level.",
            {"gallery_id": _s("Gallery id."), "lux": _i("New light level in lux.", 10, 1000)}, ["gallery_id", "lux"]),
        _fn("file_condition_report", "File a condition report for an object.",
            {"object_id": _s("Object id."), "date": _s("Date of the examination, YYYY-MM-DD."),
             "condition": _s("Overall condition.", _CONDITIONS), "examiner_id": _s("Staff id of the conservator."),
             "issues": _a("Optional list of issues found.",
                          _o({"area": _s("Where on the object."), "issue": _s("Kind of issue.", _ISSUES),
                              "severity": _s("Severity.", _SEVERITY)}, ["area", "issue", "severity"]), 1, 10)},
            ["object_id", "date", "condition", "examiner_id"]),
        _fn("send_to_conservation", "Send an object (from display or store) to the conservation studio.",
            {"object_id": _s("Object id."), "treatment": _s("Treatment needed.", _TREATMENTS),
             "priority": _s("Priority.", ["routine", "urgent"]),
             "estimated_days": _i("Estimated length of the treatment in days.", 1, 365)},
            ["object_id", "treatment", "priority", "estimated_days"]),
        _fn("complete_conservation", "Finish an object's open treatment and return it from the studio to a store.",
            {"object_id": _s("Object id."), "store_id": _s("Store id, e.g. ST-2.")}, ["object_id", "store_id"]),
        _fn("create_loan", "Create a draft outgoing loan.",
            {"partner": _s("Borrowing institution."), "object_ids": _a("Objects lent.", _s("Object id."), 1, 20),
             "start": _s("First day of the loan, YYYY-MM-DD."), "end": _s("Last day of the loan, YYYY-MM-DD."),
             "courier": _b("A museum courier accompanies the objects (default false)."),
             "max_lux": _i("Optional maximum light level at the borrower, in lux.", 10, 1000)},
            ["partner", "object_ids", "start", "end"]),
        _fn("update_loan", "Change a draft loan: add or remove objects, courier, maximum light level.",
            {"loan_id": _s("Loan id, e.g. LN-405."), "add_objects": _a("Objects to add.", _s("Object id."), 1, 20),
             "remove_objects": _a("Objects to remove.", _s("Object id."), 1, 20),
             "courier": _b("Courier required."), "max_lux": _i("Maximum light level in lux.", 10, 1000)},
            ["loan_id"]),
        _fn("approve_loan", "Approve a draft loan after the policy checks.", {"loan_id": _s("Loan id.")}, ["loan_id"]),
        _fn("cancel_loan", "Cancel a draft or approved loan.",
            {"loan_id": _s("Loan id."), "reason": _s("Why.", _LOAN_CANCEL)}, ["loan_id", "reason"]),
    ]

    def _obj(self, oid):
        return self._get("objects", oid, "object")

    def _limits(self, o):
        p = self.state["policy"]
        return p["max_lux"][o["sensitivity"]], p["annual_budget"][o["sensitivity"]]

    def _store_for(self, o):
        return self.state["policy"]["store_for"][o["sensitivity"]]

    def _days(self, until):
        return _days_between(self.state["today"], until) + 1

    def _approved_loans(self, oid, exclude=""):
        return sorted(k for k, l in self.state["loans"].items()
                      if k != exclude and l["status"] == "approved" and oid in l["objects"])

    def _displayed(self, gid):
        return sorted(k for k, o in self.state["objects"].items() if o["status"] == "on_display" and o["location"] == gid)

    def _obj_row(self, k):
        o = self.state["objects"][k]
        mx, bud = self._limits(o)
        return dict(object_id=k, max_lux=mx, annual_budget_lux_hours=bud,
                    budget_remaining=(bud - o["lux_hours_used"]) if bud else "unlimited", **o)

    def _t_get_collections_policy(self, a):
        p = self.state["policy"]
        return {"today": self.state["today"], "sensitivity_by_type": p["sensitivity"], "max_lux": p["max_lux"],
                "annual_budget_lux_hours": p["annual_budget"], "store_for_sensitivity": p["store_for"],
                "report_valid_days": p["report_valid_days"],
                "staff": [dict(staff_id=k, **s) for k, s in sorted(self.state["staff"].items())],
                "stores": self.state["stores"],
                "rules": [
                    "An object may go on display only in an open gallery with room (capacity), whose light level "
                    "does not exceed the object's max_lux, and only if it is not in poor condition and not "
                    "promised to an approved loan starting on or before its display end date.",
                    "Light budget: objects with an annual budget may go on display only if gallery lux x the "
                    "gallery's hours per day x the number of display days (today to display_until, both included) "
                    "fits within the budget remaining this year. 0 means no budget limit.",
                    "Objects in store must be in the store for their sensitivity (store_for_sensitivity).",
                    "A gallery's light level may be lowered at any time; it may be raised only if it stays within "
                    "every displayed object's max_lux and light budget.",
                    "Condition reports are filed by conservators, dated today or earlier and not before the "
                    "object's last report. A severe issue means the condition is poor; a moderate issue means it "
                    "cannot be good.",
                    "Objects on an approved loan cannot be sent to conservation. An object leaves conservation "
                    "only after a condition report dated on or after the treatment's start.",
                    "Loans: draft loans can be edited; poor objects cannot be added. To approve, every object needs "
                    "a condition report within report_valid_days before today and must not be in poor condition "
                    "or on another approved loan with overlapping dates; a loan with any high-sensitivity object "
                    "needs a courier and max_lux at most 50; otherwise max_lux must be set and at most 200.",
                ]}

    def _t_search_objects(self, a):
        q = a.get("query", "").strip().lower()
        if "query" in a and not q:
            raise _Rule("query must not be empty")
        rows = [k for k, o in sorted(self.state["objects"].items())
                if (q in o["title"].lower() or q in o["artist"].lower()) and a.get("type", o["type"]) == o["type"]
                and a.get("location", o["location"]) == o["location"] and a.get("status", o["status"]) == o["status"]]
        meta, rows = _paged(rows, a.get("page", 1), 10)
        meta["objects"] = [self._obj_row(k) for k in rows]
        return meta

    def _t_get_object(self, a):
        k = a["object_id"]
        self._obj(k)
        row = self._obj_row(k)
        row["condition_reports"] = [dict(report_id=r, **x) for r, x in sorted(self.state["reports"].items())
                                    if x["object"] == k]
        row["loans"] = [dict(loan_id=l, **x) for l, x in sorted(self.state["loans"].items()) if k in x["objects"]]
        row["treatments"] = [dict(treatment_id=t, **x) for t, x in sorted(self.state["treatments"].items())
                             if x["object"] == k]
        return row

    def _t_list_galleries(self, a):
        return {"galleries": [dict(gallery_id=k, on_display=self._displayed(k), **g)
                              for k, g in sorted(self.state["galleries"].items())]}

    def _t_list_loans(self, a):
        return {"loans": [dict(loan_id=k, **l) for k, l in sorted(self.state["loans"].items())
                          if a.get("status", l["status"]) == l["status"]]}

    def _t_move_object(self, a):
        k, to = a["object_id"], a["to_location"]
        o = self._obj(k)
        if o["status"] not in ("on_display", "in_store"):
            raise _Rule("object %s is %s" % (k, o["status"]))
        if to in self.state["galleries"]:
            g = self.state["galleries"][to]
            if "display_until" not in a:
                raise _Rule("display_until is required when moving to a gallery")
            until = _pdate(a["display_until"], "display_until").isoformat()
            if until < self.state["today"]:
                raise _Rule("display_until is in the past")
            if (o["location"], o["display_until"]) == (to, until):
                raise _Rule("object %s is already on display there until %s" % (k, until))
            if g["status"] != "open":
                raise _Rule("gallery %s (%s) is closed" % (to, g["name"]))
            if o["condition"] == "poor":
                raise _Rule("object %s is in poor condition and cannot go on display" % k)
            mx, bud = self._limits(o)
            if g["lux"] > mx:
                raise _Rule("%s is lit at %d lux; object %s (%s sensitivity) allows at most %d lux" % (
                    to, g["lux"], k, o["sensitivity"], mx))
            if o["location"] != to and len(self._displayed(to)) >= g["capacity"]:
                raise _Rule("gallery %s is full (%d objects)" % (to, g["capacity"]))
            if bud:
                need = g["lux"] * g["hours_per_day"] * self._days(until)
                if need > bud - o["lux_hours_used"]:
                    raise _Rule("displaying %s in %s until %s needs %d lux-hours; only %d remain this year" % (
                        k, to, until, need, bud - o["lux_hours_used"]))
            for l in self._approved_loans(k):
                if self.state["loans"][l]["start"] <= until:
                    raise _Rule("object %s is promised to approved loan %s starting %s" % (
                        k, l, self.state["loans"][l]["start"]))
            o.update(location=to, status="on_display", display_until=until)
        elif to in self.state["stores"]:
            if "display_until" in a:
                raise _Rule("display_until is only used for galleries")
            if o["location"] == to:
                raise _Rule("object %s is already in %s" % (k, to))
            if self._store_for(o) != to:
                raise _Rule("%s-sensitivity objects are kept in %s" % (o["sensitivity"], self._store_for(o)))
            o.update(location=to, status="in_store", display_until="")
        else:
            raise _Rule("unknown location '%s' (galleries: %s; stores: %s)" % (
                to[:20], ", ".join(sorted(self.state["galleries"])), ", ".join(sorted(self.state["stores"]))))
        return {"ok": True, "object": self._obj_row(k)}

    def _t_set_gallery_lux(self, a):
        gid = a["gallery_id"]
        g = self._get("galleries", gid, "gallery")
        if a["lux"] == g["lux"]:
            raise _Rule("gallery %s is already at %d lux" % (gid, g["lux"]))
        if a["lux"] > g["lux"]:
            for k in self._displayed(gid):
                o = self.state["objects"][k]
                mx, bud = self._limits(o)
                if a["lux"] > mx:
                    raise _Rule("object %s on display allows at most %d lux" % (k, mx))
                if bud and a["lux"] * g["hours_per_day"] * self._days(o["display_until"]) > bud - o["lux_hours_used"]:
                    raise _Rule("at %d lux object %s would exceed its light budget before %s" % (
                        a["lux"], k, o["display_until"]))
        g["lux"] = a["lux"]
        return {"ok": True, "gallery_id": gid, "lux": g["lux"]}

    def _t_file_condition_report(self, a):
        k = a["object_id"]
        o = self._obj(k)
        e = self._get("staff", a["examiner_id"], "staff member")
        if e["role"] != "conservator":
            raise _Rule("%s (%s) is a %s; condition reports are filed by conservators" % (
                a["examiner_id"], e["name"], e["role"]))
        if o["status"] == "on_loan":
            raise _Rule("object %s is out on loan" % k)
        d = _pdate(a["date"]).isoformat()
        if d > self.state["today"]:
            raise _Rule("the date is in the future (today is %s)" % self.state["today"])
        if o["last_report"] and d < o["last_report"]:
            raise _Rule("object %s already has a report dated %s" % (k, o["last_report"]))
        issues = []
        for x in a.get("issues", []):
            issues.append({"area": _text(x["area"], "area", 60), "issue": x["issue"], "severity": x["severity"]})
        sev = set(x["severity"] for x in issues)
        if "severe" in sev and a["condition"] != "poor":
            raise _Rule("a severe issue means the condition is poor")
        if "moderate" in sev and a["condition"] == "good":
            raise _Rule("with a moderate issue the condition cannot be good")
        n = self._new_id("next_report", "CR")
        self.state["reports"][n] = {"object": k, "date": d, "condition": a["condition"], "examiner": a["examiner_id"],
                                    "issues": sorted(issues, key=lambda x: (x["issue"], x["severity"], x["area"]))}
        o["condition"], o["last_report"] = a["condition"], d
        return {"ok": True, "report_id": n, "object": self._obj_row(k)}

    def _t_send_to_conservation(self, a):
        k = a["object_id"]
        o = self._obj(k)
        if o["status"] not in ("on_display", "in_store"):
            raise _Rule("object %s is %s" % (k, o["status"]))
        loans = self._approved_loans(k)
        if loans:
            raise _Rule("object %s is on approved loan %s" % (k, loans[0]))
        n = self._new_id("next_treatment", "TR")
        self.state["treatments"][n] = {"object": k, "treatment": a["treatment"], "priority": a["priority"],
                                       "estimated_days": a["estimated_days"], "started": self.state["today"],
                                       "status": "open", "finished": ""}
        o.update(status="in_conservation", location="", display_until="")
        return {"ok": True, "treatment_id": n, "object": self._obj_row(k)}

    def _t_complete_conservation(self, a):
        k = a["object_id"]
        o = self._obj(k)
        if o["status"] != "in_conservation":
            raise _Rule("object %s is %s, not in conservation" % (k, o["status"]))
        tr = [t for t, x in sorted(self.state["treatments"].items()) if x["object"] == k and x["status"] == "open"][0]
        t = self.state["treatments"][tr]
        if not o["last_report"] or o["last_report"] < t["started"]:
            raise _Rule("object %s needs a condition report dated on or after %s (start of %s) before it leaves "
                        "conservation" % (k, t["started"], tr))
        self._get("stores", a["store_id"], "store")
        if self._store_for(o) != a["store_id"]:
            raise _Rule("%s-sensitivity objects are kept in %s" % (o["sensitivity"], self._store_for(o)))
        t["status"], t["finished"] = "done", self.state["today"]
        o.update(status="in_store", location=a["store_id"])
        return {"ok": True, "object": self._obj_row(k)}

    def _loan_objects_ok(self, ids):
        if len(set(ids)) != len(ids):
            raise _Rule("an object is listed twice")
        for k in ids:
            o = self._obj(k)
            if o["condition"] == "poor":
                raise _Rule("object %s is in poor condition" % k)

    def _t_create_loan(self, a):
        partner = _text(a["partner"], "partner")
        self._loan_objects_ok(a["object_ids"])
        s, e = _pdate(a["start"], "start"), _pdate(a["end"], "end")
        if s <= self._today():
            raise _Rule("the loan must start after today")
        if e <= s:
            raise _Rule("the loan must end after it starts")
        if (e - s).days > 365:
            raise _Rule("loans last at most one year")
        n = self._new_id("next_loan", "LN")
        self.state["loans"][n] = {"partner": partner, "objects": sorted(a["object_ids"]), "start": s.isoformat(),
                                  "end": e.isoformat(), "status": "draft", "courier": a.get("courier", False),
                                  "max_lux": a.get("max_lux", 0), "cancel_reason": ""}
        return {"ok": True, "loan_id": n, "loan": self.state["loans"][n]}

    def _t_update_loan(self, a):
        k = a["loan_id"]
        l = self._get("loans", k, "loan")
        if l["status"] != "draft":
            raise _Rule("loan %s is %s; only draft loans can be changed" % (k, l["status"]))
        if len(a) < 2:
            raise _Rule("nothing to change")
        objs = list(l["objects"])
        for x in a.get("remove_objects", []):
            if x not in objs:
                raise _Rule("object %s is not on loan %s" % (x[:20], k))
            objs.remove(x)
        add = a.get("add_objects", [])
        self._loan_objects_ok(add)
        for x in add:
            if x in objs:
                raise _Rule("object %s is already on loan %s" % (x, k))
            objs.append(x)
        if not objs:
            raise _Rule("a loan needs at least one object")
        l["objects"] = sorted(objs)
        if "courier" in a:
            l["courier"] = a["courier"]
        if "max_lux" in a:
            l["max_lux"] = a["max_lux"]
        return {"ok": True, "loan": dict(loan_id=k, **l)}

    def _t_approve_loan(self, a):
        k = a["loan_id"]
        l = self._get("loans", k, "loan")
        if l["status"] != "draft":
            raise _Rule("loan %s is %s" % (k, l["status"]))
        valid_from = _plus_days(self.state["today"], -self.state["policy"]["report_valid_days"])
        high = False
        for oid in l["objects"]:
            o = self.state["objects"][oid]
            if not o["last_report"] or o["last_report"] < valid_from:
                raise _Rule("object %s needs a condition report dated on or after %s (latest: %s)" % (
                    oid, valid_from, o["last_report"] or "none"))
            if o["condition"] == "poor":
                raise _Rule("object %s is in poor condition" % oid)
            for other in self._approved_loans(oid, exclude=k):
                x = self.state["loans"][other]
                if x["start"] <= l["end"] and l["start"] <= x["end"]:
                    raise _Rule("object %s is on approved loan %s for overlapping dates" % (oid, other))
            high = high or o["sensitivity"] == "high"
        if high and (not l["courier"] or not l["max_lux"] or l["max_lux"] > 50):
            raise _Rule("a loan with high-sensitivity objects needs a courier and max_lux of at most 50")
        if not high and (not l["max_lux"] or l["max_lux"] > 200):
            raise _Rule("max_lux must be set, at most 200")
        l["status"] = "approved"
        return {"ok": True, "loan": dict(loan_id=k, **l)}

    def _t_cancel_loan(self, a):
        k = a["loan_id"]
        l = self._get("loans", k, "loan")
        if l["status"] not in ("draft", "approved"):
            raise _Rule("loan %s is %s" % (k, l["status"]))
        l["status"], l["cancel_reason"] = "cancelled", a["reason"]
        return {"ok": True, "loan": dict(loan_id=k, **l)}


def _museum():
    sens = {"painting": "medium", "works_on_paper": "high", "photograph": "high", "textile": "high",
            "furniture": "medium", "sculpture": "low", "ceramic": "low", "metalwork": "low"}
    staff = dict((k, {"name": n, "role": r}) for k, n, r in [
        ("SF-01", "Ana Ruiz", "registrar"), ("SF-02", "Dr Ines Moreau", "conservator"),
        ("SF-03", "Kwame Asante", "conservator"), ("SF-04", "Julia Berg", "curator")])
    galleries = dict((k, {"name": n, "lux": lx, "hours_per_day": h, "capacity": c, "status": st})
                     for k, n, lx, h, c, st in [
                         ("GL-1", "Main Hall", 200, 8, 10, "open"), ("GL-2", "Prints Room", 80, 8, 4, "open"),
                         ("GL-3", "Textile Gallery", 50, 6, 6, "open"), ("GL-4", "Ceramics Court", 300, 10, 12, "open"),
                         ("GL-5", "Costume Room", 50, 8, 5, "open"), ("GL-6", "East Wing", 150, 8, 6, "open"),
                         ("GL-7", "Garden Room", 120, 8, 3, "closed")])
    stores = {"ST-1": "General store", "ST-2": "Paper and textile store"}
    objects = {}
    for k, title, artist, typ, loc, status, until, cond, rep, used in [
        ("OB-1001", "Study of Gulls", "M. Ferrand", "works_on_paper", "GL-2", "on_display", "2026-12-31", "good", "2026-09-10", 60000),
        ("OB-1002", "Pier Workers", "L. Haines", "photograph", "GL-2", "on_display", "2027-01-15", "good", "2026-09-10", 30000),
        ("OB-1003", "Winter Lane", "M. Ferrand", "works_on_paper", "GL-2", "on_display", "2026-11-30", "good", "2026-09-10", 20000),
        ("OB-1004", "The Ferry Crossing", "A. Okoye", "works_on_paper", "GL-2", "on_display", "2027-03-01", "good", "2026-09-10", 10000),
        ("OB-1101", "Harbour at Dusk", "E. Lindqvist", "painting", "GL-1", "on_display", "2027-06-30", "good", "2026-09-01", 300000),
        ("OB-1104", "Portrait of a Weaver", "R. Calder", "painting", "GL-1", "on_display", "2027-03-31", "good", "2026-08-20", 150000),
        ("OB-1105", "Seated Figure", "H. Moran", "sculpture", "GL-1", "on_display", "2027-01-31", "good", "2026-07-01", 0),
        ("OB-1106", "Standing Figure", "H. Moran", "sculpture", "ST-1", "in_store", "", "good", "2026-10-02", 0),
        ("OB-1107", "Dockworkers, 1931", "L. Haines", "photograph", "GL-1", "on_display", "2027-04-30", "good", "2026-10-10", 100000),
        ("OB-1201", "Orchard in May", "E. Lindqvist", "painting", "ST-1", "in_store", "", "good", "2026-10-05", 90000),
        ("OB-1202", "Orchard in October", "E. Lindqvist", "painting", "ST-1", "in_store", "", "good", "2026-10-05", 85000),
        ("OB-1203", "Orchard Study", "E. Lindqvist", "painting", "ST-1", "in_store", "", "fair", "2026-10-05", 40000),
        ("OB-1207", "Cherry Blossom", "E. Lindqvist", "painting", "ST-1", "in_store", "", "good", "2026-10-05", 70000),
        ("OB-1210", "Mill Pond", "J. Archer", "painting", "ST-1", "in_store", "", "good", "2026-09-15", 10000),
        ("OB-1211", "Mill Race", "J. Archer", "painting", "ST-1", "in_store", "", "good", "2026-09-15", 12000),
        ("OB-1212", "Mill House", "J. Archer", "painting", "ST-1", "in_store", "", "good", "2026-09-15", 9000),
        ("OB-1301", "Embroidered Wedding Coat", "unknown", "textile", "ST-2", "in_store", "", "fair", "2026-06-12", 20000),
        ("OB-1302", "Indigo Quilt", "unknown", "textile", "GL-3", "on_display", "2027-01-31", "good", "2026-08-01", 50000),
        ("OB-1303", "Silk Banner", "unknown", "textile", "GL-3", "on_display", "2027-02-28", "good", "2026-08-01", 40000),
        ("OB-1305", "Child's Christening Gown", "unknown", "textile", "", "in_conservation", "", "fair", "2026-09-28", 100000),
        ("OB-1310", "Riding Habit", "unknown", "textile", "GL-5", "on_display", "2027-01-10", "good", "2026-08-15", 70000),
        ("OB-1311", "Evening Dress", "unknown", "textile", "GL-5", "on_display", "2027-02-14", "good", "2026-08-15", 60000),
        ("OB-1401", "Celadon Bowl", "unknown", "ceramic", "GL-4", "on_display", "2027-12-31", "good", "2026-05-05", 0),
        ("OB-1402", "Bronze Mirror", "unknown", "metalwork", "GL-4", "on_display", "2027-12-31", "good", "2026-05-05", 0),
        ("OB-1410", "Lustre Jug", "unknown", "ceramic", "ST-1", "in_store", "", "good", "2026-10-30", 0),
        ("OB-1501", "Tidal Study IV", "A. Okoye", "works_on_paper", "ST-2", "in_store", "", "good", "2026-11-02", 0),
        ("OB-1601", "Fishing Fleet", "E. Lindqvist", "painting", "GL-6", "on_display", "2027-03-31", "good", "2026-09-01", 400000),
        ("OB-1602", "Torso", "H. Moran", "sculpture", "GL-6", "on_display", "2027-02-28", "good", "2026-09-01", 0),
        ("OB-1603", "Writing Desk", "unknown", "furniture", "GL-6", "on_display", "2026-12-20", "good", "2026-09-01", 100000),
        ("OB-1701", "Shoreline Sketch", "M. Ferrand", "works_on_paper", "ST-2", "in_store", "", "good", "2026-05-01", 15000),
        ("OB-1702", "Shoreline at Noon", "M. Ferrand", "painting", "ST-1", "in_store", "", "good", "2026-10-01", 50000),
        ("OB-1703", "Shoreline in Rain", "M. Ferrand", "painting", "ST-1", "in_store", "", "poor", "2026-09-20", 50000),
        ("OB-1801", "Lighthouse", "E. Lindqvist", "painting", "", "on_loan", "", "good", "2026-08-20", 0),
    ]:
        objects[k] = {"title": title, "artist": artist, "type": typ, "sensitivity": sens[typ], "location": loc,
                      "status": status, "display_until": until, "condition": cond, "last_report": rep,
                      "lux_hours_used": used}
    loans = {}
    for k, partner, objs, start, end, status, courier, mlux in [
        ("LN-399", "Harbourside Arts Centre", ["OB-1801"], "2026-09-01", "2027-01-31", "active", True, 200),
        ("LN-405", "Valley Heritage Museum", ["OB-1210", "OB-1212"], "2027-04-01", "2027-07-31", "draft", False, 150),
        ("LN-406", "Kingsport Gallery", ["OB-1203"], "2027-01-15", "2027-03-15", "approved", False, 200),
        ("LN-410", "Northern Art Trust", ["OB-1701", "OB-1702", "OB-1703"], "2027-02-01", "2027-04-30", "draft",
         False, 0),
        ("LN-420", "Riverbend Museum", ["OB-1105", "OB-1106"], "2027-02-15", "2027-05-31", "approved", True, 200),
    ]:
        loans[k] = {"partner": partner, "objects": objs, "start": start, "end": end, "status": status,
                    "courier": courier, "max_lux": mlux, "cancel_reason": ""}
    st = {"today": "2026-11-13",
          "policy": {"sensitivity": sens, "max_lux": {"high": 50, "medium": 200, "low": 300},
                     "annual_budget": {"high": 120000, "medium": 480000, "low": 0},
                     "store_for": {"high": "ST-2", "medium": "ST-1", "low": "ST-1"}, "report_valid_days": 90},
          "staff": staff, "galleries": galleries, "stores": stores, "objects": objects, "loans": loans,
          "reports": {"CR-900": {"object": "OB-1703", "date": "2026-09-20", "condition": "poor", "examiner": "SF-02",
                                 "issues": [{"area": "lower canvas", "issue": "flaking", "severity": "severe"}]},
                      "CR-901": {"object": "OB-1305", "date": "2026-09-28", "condition": "fair", "examiner": "SF-03",
                                 "issues": [{"area": "hem", "issue": "insect_damage", "severity": "moderate"}]}},
          "treatments": {"TR-1": {"object": "OB-1305", "treatment": "pest_treatment", "priority": "routine",
                                  "estimated_days": 45, "started": "2026-10-01", "status": "open", "finished": ""}},
          "next_report": 950, "next_treatment": 10, "next_loan": 500}
    for k, o in objects.items():
        assert (o["status"] == "on_display") == (o["location"] in galleries), k
        assert (o["status"] == "in_store") == (o["location"] in stores), k
    return st


def _issues(*want):
    """Issues list predicate: (issue, severity) pairs must match exactly; the free-text area is not graded."""
    want = sorted(want)

    def ok(v):
        return isinstance(v, list) and len(v) == len(want) and all(isinstance(x, dict) and x.get("area") for x in v) \
            and sorted((x.get("issue"), x.get("severity")) for x in v) == want
    return ok


def _cr(obj, date, cond, examiner, *issues):
    return {"object": obj, "date": date, "condition": cond, "examiner": examiner, "issues": _issues(*issues)}


_MT = "2026-11-13"   # museum "today"

# ---- single-call S25-S30
_add("S25", "museum", _museum(),
     "Dr Ines Moreau (SF-02) examined the print OB-1004, The Ferry Crossing, in the Prints Room this morning, 13 "
     "November 2026. Overall condition: fair. She noted two issues - slight flaking of the ink at the upper left "
     "corner, and a moderate water stain along the bottom edge. Please file her report.",
     [["file_condition_report", {"object_id": "OB-1004", "date": "2026-11-13", "condition": "fair",
                                 "examiner_id": "SF-02", "issues": [
                                     {"area": "upper left corner", "issue": "flaking", "severity": "minor"},
                                     {"area": "bottom edge", "issue": "stain", "severity": "moderate"}]}]],
     [["file_condition_report", {"object_id": "OB-1004", "date": "2026-11-13", "condition": "fair",
                                 "examiner_id": "SF-02", "issues": [
                                     {"area": "upper left corner", "issue": "flaking", "severity": "moderate"},
                                     {"area": "bottom edge", "issue": "stain", "severity": "moderate"}]}]],
     lambda s, i: _final(s, i, changes={"objects": {"OB-1004": {"condition": "fair", "last_report": _MT}}},
                         new={"reports": [_cr("OB-1004", _MT, "fair", "SF-02", ("flaking", "minor"),
                                              ("stain", "moderate"))]}),
     "ONDX", mentions=["Ines Moreau", "The Ferry Crossing", "Prints Room"])

_add("S26", "museum", _museum(),
     "Put the Lustre Jug (OB-1410, currently in the general store) on display in the Ceramics Court, GL-4, from "
     "today. It should stay out until exactly six weeks after the winter exhibition opens on 20 November 2026, so "
     "set its last display day to that date.",
     [["move_object", {"object_id": "OB-1410", "to_location": "GL-4", "display_until": "2027-01-01"}]],
     [["move_object", {"object_id": "OB-1410", "to_location": "GL-4", "display_until": "2026-12-25"}]],
     lambda s, i: _final(s, i, changes={"objects": {"OB-1410": {"location": "GL-4", "status": "on_display",
                                                                "display_until": "2027-01-01"}}}),
     "DT", mentions=["Lustre Jug", "Ceramics Court"])

_add("S27", "museum", _museum(),
     "The lighting designer wants the East Wing (GL-6) dimmed to 11 foot-candles. Our system works in lux: use "
     "1 foot-candle = 10.764 lux and round to the nearest whole lux. (The Main Hall stays at 200 lux and the "
     "Prints Room at 80.)",
     [["set_gallery_lux", {"gallery_id": "GL-6", "lux": 118}]],
     [["set_gallery_lux", {"gallery_id": "GL-6", "lux": 110}]],
     lambda s, i: _final(s, i, changes={"galleries": {"GL-6": {"lux": 118}}}),
     "UX", mentions=["East Wing", "Main Hall", "Prints Room"])

_add("S28", "museum", _museum(),
     "Set up a draft loan to the Kunsthalle Nord for their Lindqvist show: Orchard in May (OB-1201), Orchard in "
     "October (OB-1202) and Cherry Blossom (OB-1207) - not Orchard Study, which is still being researched. The "
     "loan runs from 1 March to 30 June 2027. One of our couriers travels with the works, and they may be lit at "
     "no more than 150 lux.",
     [["create_loan", {"partner": "Kunsthalle Nord", "object_ids": ["OB-1201", "OB-1202", "OB-1207"],
                       "start": "2027-03-01", "end": "2027-06-30", "courier": True, "max_lux": 150}]],
     [["create_loan", {"partner": "Kunsthalle Nord", "object_ids": ["OB-1201", "OB-1202", "OB-1207"],
                       "start": "2027-03-01", "end": "2027-06-30", "max_lux": 150}]],
     lambda s, i: _final(s, i, new={"loans": [{"partner": "Kunsthalle Nord", "objects": ["OB-1201", "OB-1202", "OB-1207"],
                                               "start": "2027-03-01", "end": "2027-06-30", "status": "draft",
                                               "courier": True, "max_lux": 150, "cancel_reason": ""}]}),
     "XPD", mentions=["Orchard in May", "Orchard in October", "Cherry Blossom", "Orchard Study", "Lindqvist"],
     creates=["Kunsthalle Nord"])

_add("S29", "museum", _museum(),
     "Bad news from the paper and textile store: live clothes-moth larvae in the folds of the Embroidered Wedding "
     "Coat (OB-1301). It has to go to the conservation studio straight away, ahead of everything else, to have "
     "the insects dealt with by freezing. The studio expects the job to take three weeks.",
     [["send_to_conservation", {"object_id": "OB-1301", "treatment": "pest_treatment", "priority": "urgent",
                                "estimated_days": 21}]],
     [["send_to_conservation", {"object_id": "OB-1301", "treatment": "cleaning", "priority": "urgent",
                                "estimated_days": 21}]],
     lambda s, i: _final(s, i, changes={"objects": {"OB-1301": {"status": "in_conservation", "location": ""}}},
                         new={"treatments": [{"object": "OB-1301", "treatment": "pest_treatment", "priority": "urgent",
                                              "estimated_days": 21, "started": _MT, "status": "open",
                                              "finished": ""}]}),
     "NUT", mentions=["Embroidered Wedding Coat"])

_add("S30", "museum", _museum(),
     "On the draft loan LN-405 to the Valley Heritage Museum, swap Mill Pond (OB-1210) for Mill Race (OB-1211) - "
     "the borrower prefers the second Archer canvas. Mill House (OB-1212) stays on the loan, and the dates and "
     "lighting stay as they are.",
     [["update_loan", {"loan_id": "LN-405", "add_objects": ["OB-1211"], "remove_objects": ["OB-1210"]}]],
     [["update_loan", {"loan_id": "LN-405", "add_objects": ["OB-1211"]}]],
     lambda s, i: _final(s, i, changes={"loans": {"LN-405": {"objects": ["OB-1211", "OB-1212"]}}}),
     "TX", mentions=["Valley Heritage Museum", "Mill Pond", "Mill Race", "Mill House"])

# ---- multi-step M25-M30
_add("M25", "museum", _museum(),
     "Please install the new print Tidal Study IV in the Prints Room, on display until 28 February 2027. If the "
     "room is lit too brightly for it, first bring the room down to the brightest level the print allows. If the "
     "room is full, the object there whose display ends soonest goes back to its store to make space. Nothing "
     "else should change.",
     [["get_collections_policy", {}], ["search_objects", {"query": "Tidal Study"}], ["list_galleries", {}],
      ["move_object", {"object_id": "OB-1501", "to_location": "GL-2", "display_until": "2027-02-28"}],
      ["set_gallery_lux", {"gallery_id": "GL-2", "lux": 50}],
      ["move_object", {"object_id": "OB-1501", "to_location": "GL-2", "display_until": "2027-02-28"}],
      ["search_objects", {"location": "GL-2"}],
      ["move_object", {"object_id": "OB-1003", "to_location": "ST-1"}],
      ["move_object", {"object_id": "OB-1003", "to_location": "ST-2"}],
      ["move_object", {"object_id": "OB-1501", "to_location": "GL-2", "display_until": "2027-02-28"}],
      ["list_galleries", {}]],
     [["set_gallery_lux", {"gallery_id": "GL-2", "lux": 40}],
      ["move_object", {"object_id": "OB-1001", "to_location": "ST-2"}],
      ["move_object", {"object_id": "OB-1501", "to_location": "GL-2", "display_until": "2027-02-28"}]],
     lambda s, i: _final(s, i, changes={
         "galleries": {"GL-2": {"lux": 50}},
         "objects": {"OB-1003": {"location": "ST-2", "status": "in_store", "display_until": ""},
                     "OB-1501": {"location": "GL-2", "status": "on_display", "display_until": "2027-02-28"}}}),
     "abcf", ref_errors=[3, 5, 7], mentions=["Tidal Study IV", "Prints Room"])

_add("M26", "museum", _museum(),
     "Get the draft loan to the Northern Art Trust approved. Any object on it without a recent enough condition "
     "report was examined this morning by Dr Ines Moreau and found in good condition with no issues - record "
     "that. Objects in poor condition come off the loan. Set whatever courier and lighting terms the policy "
     "requires, choosing the brightest light level it permits. Nothing else should change.",
     [["get_collections_policy", {}], ["list_loans", {"status": "draft"}],
      ["approve_loan", {"loan_id": "LN-410"}],
      ["file_condition_report", {"object_id": "OB-1701", "date": "2026-11-13", "condition": "good",
                                 "examiner_id": "SF-02"}],
      ["approve_loan", {"loan_id": "LN-410"}],
      ["update_loan", {"loan_id": "LN-410", "remove_objects": ["OB-1703"], "courier": True, "max_lux": 50}],
      ["approve_loan", {"loan_id": "LN-410"}],
      ["list_loans", {"status": "approved"}]],
     [["file_condition_report", {"object_id": "OB-1701", "date": "2026-11-13", "condition": "good",
                                 "examiner_id": "SF-02"}],
      ["update_loan", {"loan_id": "LN-410", "remove_objects": ["OB-1701", "OB-1703"], "max_lux": 200}],
      ["approve_loan", {"loan_id": "LN-410"}]],
     lambda s, i: _final(s, i, changes={
         "loans": {"LN-410": {"objects": ["OB-1701", "OB-1702"], "courier": True, "max_lux": 50, "status": "approved"}},
         "objects": {"OB-1701": {"last_report": _MT}}},
         new={"reports": [{"object": "OB-1701", "date": _MT, "condition": "good", "examiner": "SF-02", "issues": []}]}),
     "abcf", ref_errors=[2, 4], mentions=["Northern Art Trust", "Ines Moreau"])

_add("M27", "museum", _museum(),
     "The Child's Christening Gown is finished in the conservation studio. Kwame Asante examined it today: "
     "condition fair, with minor fading on the left sleeve - file that. Then return it from the studio to the "
     "right store, and put it on display in the Costume Room until 31 January 2027 if the rules allow that; if "
     "they don't, it simply stays in store. Nothing else should change.",
     [["get_collections_policy", {}], ["search_objects", {"query": "Christening Gown"}],
      ["complete_conservation", {"object_id": "OB-1305", "store_id": "ST-2"}],
      ["file_condition_report", {"object_id": "OB-1305", "date": "2026-11-13", "condition": "fair",
                                 "examiner_id": "SF-03", "issues": [
                                     {"area": "left sleeve", "issue": "fading", "severity": "minor"}]}],
      ["complete_conservation", {"object_id": "OB-1305", "store_id": "ST-2"}],
      ["list_galleries", {}],
      ["move_object", {"object_id": "OB-1305", "to_location": "GL-5", "display_until": "2027-01-31"}],
      ["get_object", {"object_id": "OB-1305"}]],
     [["file_condition_report", {"object_id": "OB-1305", "date": "2026-11-13", "condition": "good",
                                 "examiner_id": "SF-03", "issues": [
                                     {"area": "left sleeve", "issue": "fading", "severity": "minor"}]}],
      ["complete_conservation", {"object_id": "OB-1305", "store_id": "ST-2"}],
      ["move_object", {"object_id": "OB-1305", "to_location": "GL-3", "display_until": "2027-01-31"}]],
     lambda s, i: _final(s, i, changes={
         "objects": {"OB-1305": {"status": "in_store", "location": "ST-2", "last_report": _MT}},
         "treatments": {"TR-1": {"status": "done", "finished": _MT}}},
         new={"reports": [_cr("OB-1305", _MT, "fair", "SF-03", ("fading", "minor"))]}),
     "abcef", ref_errors=[2, 6], mentions=["Kwame Asante", "Costume Room"])

_add("M28", "museum", _museum(),
     "The East Wing closes today for refurbishment. Taking its objects in order of object id, move each one, with "
     "its current display end date, to the open gallery with the lowest id where the rules allow it to be shown "
     "until then. An object that no gallery can take goes to its store. The light levels of the galleries stay "
     "as they are, and nothing else should change.",
     [["get_collections_policy", {}], ["list_galleries", {}], ["search_objects", {"location": "GL-6"}],
      ["move_object", {"object_id": "OB-1601", "to_location": "GL-1", "display_until": "2027-03-31"}],
      ["move_object", {"object_id": "OB-1601", "to_location": "GL-2", "display_until": "2027-03-31"}],
      ["move_object", {"object_id": "OB-1601", "to_location": "GL-3", "display_until": "2027-03-31"}],
      ["move_object", {"object_id": "OB-1602", "to_location": "GL-1", "display_until": "2027-02-28"}],
      ["move_object", {"object_id": "OB-1603", "to_location": "GL-1", "display_until": "2026-12-20"}],
      ["list_galleries", {}]],
     [["move_object", {"object_id": "OB-1601", "to_location": "ST-1"}],
      ["move_object", {"object_id": "OB-1602", "to_location": "GL-1", "display_until": "2027-02-28"}],
      ["move_object", {"object_id": "OB-1603", "to_location": "GL-1", "display_until": "2026-12-20"}]],
     lambda s, i: _final(s, i, changes={"objects": {
         "OB-1601": {"location": "GL-3"}, "OB-1602": {"location": "GL-1"}, "OB-1603": {"location": "GL-1"}}}),
     "abdef", ref_errors=[3, 4], mentions=["East Wing"])

_add("M29", "museum", _museum(),
     "Light audit. For every gallery whose light level is above what one of the objects on display there "
     "allows, lower the light level to the lowest maximum among the objects shown in that gallery. Then, at the "
     "new levels, any high-sensitivity object whose remaining light budget won't last until its display end date "
     "goes back to its store. Galleries that are fine stay as they are, and nothing else should change.",
     [["get_collections_policy", {}], ["list_galleries", {}], ["search_objects", {"status": "on_display"}],
      ["search_objects", {"status": "on_display", "page": 2}],
      ["set_gallery_lux", {"gallery_id": "GL-1", "lux": 50}], ["set_gallery_lux", {"gallery_id": "GL-2", "lux": 50}],
      ["list_galleries", {}],
      ["move_object", {"object_id": "OB-1107", "to_location": "ST-1"}],
      ["move_object", {"object_id": "OB-1107", "to_location": "ST-2"}]],
     [["set_gallery_lux", {"gallery_id": "GL-2", "lux": 50}],
      ["move_object", {"object_id": "OB-1107", "to_location": "ST-2"}]],
     lambda s, i: _final(s, i, changes={
         "galleries": {"GL-1": {"lux": 50}, "GL-2": {"lux": 50}},
         "objects": {"OB-1107": {"location": "ST-2", "status": "in_store", "display_until": ""}}}),
     "adefg", ref_errors=[7])

_add("M30", "museum", _museum(),
     "Kwame Asante surveyed three objects in the Main Hall today. Harbour at Dusk: good, no issues. Portrait of "
     "a Weaver: fair, a minor stain at the lower right. Seated Figure: poor, a severe crack in the base. File the "
     "three reports. Anything now in poor condition goes to the conservation studio for urgent repair, estimated "
     "at 30 days; if a loan stands in the way, cancel that loan because of the condition. Nothing else should "
     "change.",
     [["get_collections_policy", {}], ["search_objects", {"location": "GL-1"}],
      ["file_condition_report", {"object_id": "OB-1101", "date": "2026-11-13", "condition": "good",
                                 "examiner_id": "SF-03"}],
      ["file_condition_report", {"object_id": "OB-1104", "date": "2026-11-13", "condition": "fair",
                                 "examiner_id": "SF-03", "issues": [
                                     {"area": "lower right", "issue": "stain", "severity": "minor"}]}],
      ["file_condition_report", {"object_id": "OB-1105", "date": "2026-11-13", "condition": "poor",
                                 "examiner_id": "SF-03", "issues": [
                                     {"area": "base", "issue": "crack", "severity": "severe"}]}],
      ["send_to_conservation", {"object_id": "OB-1105", "treatment": "repair", "priority": "urgent",
                                "estimated_days": 30}],
      ["list_loans", {"status": "approved"}],
      ["cancel_loan", {"loan_id": "LN-420", "reason": "condition_concern"}],
      ["send_to_conservation", {"object_id": "OB-1105", "treatment": "repair", "priority": "urgent",
                                "estimated_days": 30}],
      ["get_object", {"object_id": "OB-1105"}]],
     [["file_condition_report", {"object_id": "OB-1101", "date": "2026-11-13", "condition": "good",
                                 "examiner_id": "SF-03"}],
      ["file_condition_report", {"object_id": "OB-1104", "date": "2026-11-13", "condition": "fair",
                                 "examiner_id": "SF-03", "issues": [
                                     {"area": "lower right", "issue": "stain", "severity": "minor"}]}],
      ["file_condition_report", {"object_id": "OB-1105", "date": "2026-11-13", "condition": "poor",
                                 "examiner_id": "SF-03", "issues": [
                                     {"area": "base", "issue": "crack", "severity": "severe"}]}],
      ["cancel_loan", {"loan_id": "LN-420", "reason": "partner_withdrew"}],
      ["send_to_conservation", {"object_id": "OB-1105", "treatment": "repair", "priority": "urgent",
                                "estimated_days": 30}]],
     lambda s, i: _final(s, i, changes={
         "objects": {"OB-1101": {"last_report": _MT}, "OB-1104": {"condition": "fair", "last_report": _MT},
                     "OB-1105": {"condition": "poor", "last_report": _MT, "status": "in_conservation", "location": "",
                                 "display_until": ""}},
         "loans": {"LN-420": {"status": "cancelled", "cancel_reason": "condition_concern"}}},
         new={"reports": [_cr("OB-1101", _MT, "good", "SF-03"), _cr("OB-1104", _MT, "fair", "SF-03", ("stain", "minor")),
                          _cr("OB-1105", _MT, "poor", "SF-03", ("crack", "severe"))],
              "treatments": [{"object": "OB-1105", "treatment": "repair", "priority": "urgent", "estimated_days": 30,
                              "started": _MT, "status": "open", "finished": ""}]}),
     "bcdf", ref_errors=[5], mentions=["Kwame Asante", "Main Hall", "Harbour at Dusk", "Portrait of a Weaver",
                                       "Seated Figure"], seq=[[4, 8]])

# ---- calibration C5 (single)
_add("C5", "museum", _museum(),
     "Kingsport Gallery has written to say they are formally withdrawing from the exhibition, so our approved "
     "loan LN-406 of Orchard Study to them will not go ahead. Please cancel the loan with the reason that fits. "
     "Our own dates were fine.",
     [["cancel_loan", {"loan_id": "LN-406", "reason": "partner_withdrew"}]],
     [["cancel_loan", {"loan_id": "LN-406", "reason": "schedule_conflict"}]],
     lambda s, i: _final(s, i, changes={"loans": {"LN-406": {"status": "cancelled", "cancel_reason": "partner_withdrew"}}}),
     "NT", mentions=["Kingsport Gallery", "Orchard Study"])


ENVS = {"vet": VetEnv, "warehouse": WarehouseEnv, "lab": LabEnv, "league": LeagueEnv, "museum": MuseumEnv}


def _order_key(t):
    tid = t["id"]
    return ("SMC".index(tid[0]), int(tid[1:]))


TASKS.sort(key=_order_key)

# Difficulty ingredients (documentation; _verify insists on at least two for single-call, three for multi-step).
# single-call (upper case):
#   T  the right tool must be chosen among near-duplicates (schedule vs reschedule vs follow-up, move vs return, ...)
#   U  a unit or time conversion stated in the prompt (lb/oz -> g, in -> cm, "1 h 20 min" -> minutes, ...)
#   N  an enum value that the prompt only paraphrases
#   D  an ISO date (or time) computed from a date given in the prompt
#   O  a nested array of objects
#   P  an optional parameter that must be set
#   X  numbers that must be copied exactly from a cluttered prompt
# multi-step (lower case, as in the lfm25-tools set):
#   a  multi-way conditional decided by data that must be read and aggregated
#   b  business-rule error on the natural first attempt that needs a specific recovery
#   c  ordering dependency (the env rejects, or the outcome changes, in the wrong order)
#   d  batch over several entities picked by a compound criterion, with look-alikes that must stay untouched
#   e  computed argument that depends on two or more earlier read results
#   f  rule that is only stated by a policy read-tool / tool result, not by the prompt
#   g  explicit "nothing else may change" with a tempting side effect


# --------------------------------------------------------------------------
# self-verification
# --------------------------------------------------------------------------
def _is_err(res):
    return isinstance(res, dict) and "error" in res


def _strings(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(k)
            _strings(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, acc)
    elif isinstance(obj, str):
        acc.add(obj)


_GARBAGE = [None, "", "zzz", "/", "AP-3010; drop", -1, 0, 3, 10 ** 12, 1.5, 2.0, float("inf"), float("nan"), True,
            False, [], [None], [{}], ["a", 1], [[1]], {}, {"x": {"y": [1]}}, {"id": 1, "qty": "2"},
            [{"vaccine": "dhpp"}], [{"sku": "SKU-1", "qty": -2}], "[1, 2", "[]", "[{}]",
            "2026-13-45", "25:99", "2026-11-09", "2026-11-30", "2027-01-15", "09:00", "9:00", "é中", "null"]


def _safe_call(env, name, args, label):
    before = _clone(env.state)
    try:
        res = env.call(name, args)
    except Exception as e:  # pragma: no cover - this is what the fuzz is for
        raise AssertionError("%s: call raised %r for %r %r" % (label, e, name, args))
    json.dumps(res)
    if _is_err(res):
        assert env.state == before, "%s: state changed on error for %r %r" % (label, name, args)
    if _is_read(name):
        assert env.state == before, "%s: read tool %r changed the state" % (label, name)
    json.dumps(env.state)
    return res


def _fuzz():
    rng = random.Random(20261109)
    calls = 0
    for ename in sorted(ENVS):
        cls = ENVS[ename]
        inits = [t["init"] for t in TASKS if t["env"] == ename]
        pool = set()
        for ini in inits:
            _strings(ini, pool)
        pool = sorted(pool)
        dates = [x for x in pool if re.match(r"^\d{4}-\d{2}-\d{2}$", x)] or ["2026-11-09"]
        times = [x for x in pool if re.match(r"^\d{2}:\d{2}$", x)] or ["09:00"]
        names = [t["function"]["name"] for t in cls.TOOLS]
        env = cls(inits[0])
        for name in names + [None, 5, "nope", "", [], {}, 1.5, "cancel_appointment; drop"]:
            for args in _GARBAGE + ["{}", 7]:
                _safe_call(env, name, args, ename)
                calls += 1

        def rand_value(schema, depth=0):
            r = rng.random()
            if r < 0.25 or depth > 3:
                return rng.choice(_GARBAGE)
            t = schema.get("type")
            if t == "string":
                if "enum" in schema and rng.random() < 0.85:
                    return rng.choice(schema["enum"])
                return rng.choice(pool)
            if t == "integer":
                return rng.choice([-5, 0, 1, 2, 3, 6, 8, 9, 10, 12, 15, 20, 46, 100, 150, 1000, 5000, 19200])
            if t == "boolean":
                return rng.random() < 0.5
            if t == "array":
                return [rand_value(schema["items"], depth + 1) for _ in range(rng.randint(0, 3))]
            if t == "object":
                out = {}
                for k, sub in schema.get("properties", {}).items():
                    if k in schema.get("required", []) or rng.random() < 0.5:
                        out[k] = rand_value(sub, depth + 1)
                if rng.random() < 0.05:
                    out["bogus"] = 1
                return out
            return None

        def typed_value(key, schema):
            """A plausible value for a parameter: ids of the right family, so that many calls succeed."""
            if "enum" in schema:
                return rng.choice(schema["enum"])
            t = schema.get("type")
            if t == "boolean":
                return rng.random() < 0.5
            if t == "integer":
                lo, hi = schema.get("minimum", 1), schema.get("maximum", 5000)
                return rng.choice([x for x in [1, 2, 3, 5, 8, 10, 12, 20, 50, 120, 200, 400, 1000, 4000, 12000]
                                   if lo <= x <= hi] or [lo])
            if t == "array":
                return [typed_value(key, schema["items"]) for _ in range(rng.randint(1, 3))]
            if t == "object":
                return dict((k, typed_value(k, sub)) for k, sub in schema["properties"].items()
                            if k in schema["required"] or rng.random() < 0.4)
            if "date" in key or key in ("until", "start", "end", "due_back", "display_until"):
                return rng.choice(dates)
            if "time" in key or key == "kickoff":
                return rng.choice(times)
            return rng.choice(pool)

        ok_writes = 0
        for n in range(3000):
            if n % 150 == 0:
                env = cls(rng.choice(inits))
            tool = rng.choice(cls.TOOLS)["function"]
            if n % 2:
                args = rand_value(tool["parameters"])
            else:
                props = tool["parameters"]["properties"]
                args = dict((k, typed_value(k, sub)) for k, sub in props.items()
                            if k in tool["parameters"]["required"] or rng.random() < 0.3)
            res = _safe_call(env, tool["name"], args, ename)
            if not _is_err(res) and not _is_read(tool["name"]):
                ok_writes += 1
            calls += 1
        assert ok_writes > 0, (ename, "random fuzz never produced a successful write")
    # mutate every argument of every reference call with every garbage value, and delete each argument
    for t in TASKS:
        base = ENVS[t["env"]](t["init"])          # state just before each reference step
        for step, (name, args) in enumerate(t["reference"]):
            for key in list(args) + ["__extra__"]:
                for g in _GARBAGE:
                    env = ENVS[t["env"]](base.state)
                    bad = copy.deepcopy(args)
                    bad[key] = g
                    _safe_call(env, name, bad, t["id"])
                    calls += 1
                if isinstance(args.get(key), list):          # and every field of every nested object
                    for n, item in enumerate(args[key]):
                        if not isinstance(item, dict):
                            continue
                        for sub in list(item) + ["__extra__"]:
                            for g in _GARBAGE[::3]:
                                env = ENVS[t["env"]](base.state)
                                bad = copy.deepcopy(args)
                                bad[key][n][sub] = g
                                _safe_call(env, name, bad, t["id"])
                                calls += 1
            for key in args:
                env = ENVS[t["env"]](base.state)
                bad = copy.deepcopy(args)
                del bad[key]
                _safe_call(env, name, bad, t["id"])
                calls += 1
            base.call(name, copy.deepcopy(args))
    return calls


def _schema_ok(schema, path, depth=0):
    """No union types, no null, only simple types; arrays of scalars or of flat objects, one level deep."""
    t = schema.get("type")
    assert isinstance(t, str), (path, "type must be a single string, got %r" % (t,))
    assert t in ("string", "integer", "boolean", "array", "object"), (path, t)
    for bad in ("anyOf", "oneOf", "allOf", "nullable"):
        assert bad not in schema, (path, bad)
    if "enum" in schema:
        assert all(isinstance(x, str) and x for x in schema["enum"]), (path, "enum")
    if t == "array":
        it = schema["items"]
        assert it["type"] in ("string", "integer", "boolean", "object"), (path, "array items")
        if it["type"] == "object":
            assert depth == 0, (path, "objects may only be nested one level deep")
            assert it.get("additionalProperties") is False and it.get("required"), (path, "item object")
            for k, sub in it["properties"].items():
                assert sub["type"] in ("string", "integer", "boolean"), (path, k, "item fields must be scalars")
                _schema_ok(sub, path + "[]." + k, depth + 1)
            for r in it["required"]:
                assert r in it["properties"], (path, r)
    if t == "object":
        assert depth == 0, (path, "no nested objects outside arrays")
        for k, sub in schema.get("properties", {}).items():
            _schema_ok(sub, path + "." + k, depth)


def _no_empty(obj, where):
    if obj is None or obj == "":
        raise AssertionError("%s: reference passes None or an empty string" % (where,))
    if isinstance(obj, dict):
        for v in obj.values():
            _no_empty(v, where)
    elif isinstance(obj, list):
        for v in obj:
            _no_empty(v, where)


_ID_PATTERNS = [r"\b[A-Z]{1,5}\d{0,3}-[A-Z0-9]{1,5}\b",     # A-101, AP-3010, OR-5521, SM-0412, FX-210, OB-1044
                r"\b[A-Z]{2,5}\d{3}\b",
                r"\b[A-Z]{4,}\b"]                           # anything written in capitals


def _prompt_entities(t):
    """Every id-like token, quoted name and declared mention in the prompt must exist in init."""
    pool = set()
    _strings(t["init"], pool)
    blob = "\n".join(sorted(pool))
    vocab = set()                      # enum values of the env's tool schemas may be quoted too
    _strings(ENVS[t["env"]].TOOLS, vocab)
    prompt = t["prompt"]
    found = set()
    for pat in _ID_PATTERNS:
        for tok in re.findall(pat, prompt):
            found.add(tok)
            assert tok in pool or tok in t["creates"], (t["id"], "prompt names %r which is not in init" % tok)
    for tok in re.findall(r"(?<![A-Za-z0-9])'([^']{2,40})'(?![A-Za-z0-9])", prompt):
        found.add(tok)
        assert tok in pool or tok in vocab or tok in t["creates"], (t["id"], "prompt quotes %r which is not in init" % tok)
    for m in t["mentions"]:
        assert m in prompt, (t["id"], "declared mention %r is not in the prompt" % m)
        assert m in blob, (t["id"], "prompt mentions %r which is not in init" % m)
    for c in t["creates"]:
        assert c in prompt, (t["id"], "declared creation %r is not in the prompt" % c)
    return len(found) + len(t["mentions"])


def _alt_orders(t, rng, tries=300):
    """Replay the reference's successful WRITE calls in other orders. An order is 'valid' if no call errors;
    every valid order must satisfy check. Orders that break a declared seq group are not tried."""
    writes = [n for n, (name, _) in enumerate(t["reference"]) if not _is_read(name) and n not in t["ref_errors"]]
    orders, seen = [list(reversed(writes))], set()
    for _ in range(tries):
        p = list(writes)
        rng.shuffle(p)
        orders.append(p)
    for n in range(len(writes) - 1):       # every adjacent swap of the reference order
        p = list(writes)
        p[n], p[n + 1] = p[n + 1], p[n]
        orders.append(p)
    valid = 0
    for p in orders:
        if tuple(p) in seen or p == writes:
            continue
        seen.add(tuple(p))
        if any([x for x in p if x in g] != [x for x in writes if x in g] for g in t["seq"]):
            continue
        env = ENVS[t["env"]](t["init"])
        ok = True
        for n in p:
            name, args = t["reference"][n]
            if _is_err(env.call(name, copy.deepcopy(args))):
                ok = False
                break
        if not ok:
            continue
        valid += 1
        assert t["check"](env.state) is True, (t["id"], "valid alternative write order fails check", p)
    return valid, len(writes)


def _variants(v, pool):
    """Other well-formed values for one argument leaf (used to perturb single-call references)."""
    out = []
    if isinstance(v, bool):
        out.append(not v)
    elif isinstance(v, int):
        out += [v - 1, v + 1, v * 10, v // 10]
    elif isinstance(v, str):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            out += [_plus_days(v, -1), _plus_days(v, 1), _plus_days(v, 7)]
        elif re.match(r"^\d{2}:\d{2}$", v):
            m = _ptime(v)
            out += [_fmt_time(m + d) for d in (-60, -10, -5, 5, 10, 60, 720) if 0 <= m + d < 1440]
        else:
            m = re.match(r"^([A-Z]{1,4}-)", v)
            if m:   # other ids of the same family
                out += [x for x in pool if x.startswith(m.group(1)) and x != v][:40]
    return out


def _perturb_single(t):
    """Every well-formed perturbation of the single reference call that succeeds and leads to a different state
    must fail the check (an extra, missing or different argument is a different write)."""
    name, args = t["reference"][0]
    env = ENVS[t["env"]](t["init"])
    env.call(name, copy.deepcopy(args))
    solved = env.state
    pool = set()
    _strings(t["init"], pool)
    pool = sorted(pool)
    tool = [x["function"] for x in ENVS[t["env"]].TOOLS if x["function"]["name"] == name][0]
    cands = []

    def schema_at(path):
        sch = tool["parameters"]
        for p in path:
            sch = sch["items"] if isinstance(p, int) else sch["properties"][p]
        return sch

    def leaves(obj, path):
        if isinstance(obj, dict):
            req = schema_at(path).get("required", [])
            for k in obj:
                leaves(obj[k], path + [k])
                if k not in req:                           # leave out an optional argument
                    cands.append(("drop", path + [k], None))
        elif isinstance(obj, list):
            for n in range(len(obj)):
                leaves(obj[n], path + [n])
                if len(obj) > 1:                           # drop one element
                    cands.append(("drop", path + [n], None))
        else:
            cands.append(("set", path, obj))

    leaves(args, [])

    def enum_of(path):
        return schema_at(path).get("enum")

    tried = changed = 0
    for kind, path, val in cands:
        if kind == "drop":
            vs = [None]
        else:
            vs = list(enum_of(path) or []) + _variants(val, pool)
            vs = [x for x in vs if x != val]
        for x in vs:
            bad = copy.deepcopy(args)
            ref = bad
            for p in path[:-1]:
                ref = ref[p]
            if kind == "drop":
                del ref[path[-1]]
            else:
                ref[path[-1]] = x
            env = ENVS[t["env"]](t["init"])
            res = env.call(name, bad)
            tried += 1
            if not _is_err(res) and env.state != solved:
                changed += 1
                assert t["check"](env.state) is False, (t["id"], "perturbed call passes the check", path, x)
    # optional parameters the reference leaves out: adding any well-formed value must fail if it changes the state
    for k, sub in tool["parameters"]["properties"].items():
        if k in args or sub["type"] in ("array", "object"):
            continue
        vs = list(sub.get("enum", [])) or ([True, False] if sub["type"] == "boolean" else
                                           [1, 2, 5] if sub["type"] == "integer" else [])
        for x in vs:
            bad = copy.deepcopy(args)
            bad[k] = x
            env = ENVS[t["env"]](t["init"])
            res = env.call(name, bad)
            tried += 1
            if not _is_err(res) and env.state != solved:
                changed += 1
                assert t["check"](env.state) is False, (t["id"], "extra optional argument passes the check", k, x)
    return tried, changed


def _rereads(t):
    """Number of reference read calls, issued after a successful write, whose result differs from the result
    the same call gives on the initial state (i.e. state re-reads that matter)."""
    env = ENVS[t["env"]](t["init"])
    wrote, n = False, 0
    for k, (name, args) in enumerate(t["reference"]):
        res = env.call(name, copy.deepcopy(args))
        if _is_read(name):
            if wrote and not _is_err(res):
                fresh = ENVS[t["env"]](t["init"]).call(name, copy.deepcopy(args))
                if json.dumps(fresh, sort_keys=True) != json.dumps(res, sort_keys=True):
                    n += 1
        elif not _is_err(res):
            wrote = True
    return n


# Hand-written alternative ROUTES (different calls, not just a different order) that are equally valid
# solutions; each must satisfy check. Guards against checks that only fit the reference's route.
_ALT_ROUTES = {
    "S03": [["record_vaccinations", {"patient_id": "A-105", "date_given": "2026-11-09", "vet_id": "V-02", "vaccines": [
        {"vaccine": "kennel_cough", "batch_number": "KC-4410", "route": "intranasal"},
        {"vaccine": "lepto", "batch_number": "LP-2093", "route": "subcutaneous"},
        {"vaccine": "dhpp", "batch_number": "DH-7781", "route": "subcutaneous"}]}]],
    "S04": [["reschedule_appointment", {"appointment_id": "AP-3010", "date": "2026-11-12", "time": "11:30",
                                        "vet_id": "V-03"}]],
    "S20": [["reschedule_fixture", {"fixture_id": "FX-214", "date": "2026-11-22", "kickoff": "10:30",
                                    "venue_id": "VN-4", "pitch": 2}]],
    "S23": [["submit_team_sheet", {"fixture_id": "FX-220", "team_id": "TM-21", "captain_id": "PL-503", "players": [
        {"player_id": "PL-508", "shirt": 12, "starting": False}, {"player_id": "PL-507", "shirt": 10, "starting": True},
        {"player_id": "PL-506", "shirt": 9, "starting": True}, {"player_id": "PL-505", "shirt": 8, "starting": True},
        {"player_id": "PL-504", "shirt": 4, "starting": True}, {"player_id": "PL-503", "shirt": 3, "starting": True},
        {"player_id": "PL-502", "shirt": 2, "starting": True}, {"player_id": "PL-501", "shirt": 1, "starting": True}]}]],
    "M02": [["stop_prescription", {"prescription_id": "RX-902", "reason": "patient_deceased"}],
            ["update_patient", {"patient_id": "A-117", "status": "deceased"}],
            ["cancel_appointment", {"appointment_id": "AP-3006", "reason": "patient_deceased"}],
            ["cancel_appointment", {"appointment_id": "AP-3005", "reason": "patient_deceased"}],
            ["update_patient", {"patient_id": "A-117", "status": "deceased"}]],
    "M03": [["cancel_appointment", {"appointment_id": "AP-3032", "reason": "vet_unavailable"}],
            ["reschedule_appointment", {"appointment_id": "AP-3033", "date": "2026-11-13", "time": "11:00",
                                        "vet_id": "V-05"}],
            ["reschedule_appointment", {"appointment_id": "AP-3031", "date": "2026-11-13", "time": "09:30",
                                        "vet_id": "V-05"}],
            ["reschedule_appointment", {"appointment_id": "AP-3030", "date": "2026-11-13", "time": "09:00",
                                        "vet_id": "V-05"}]],
    "M05": [["record_vaccinations", {"patient_id": "A-115", "date_given": "2026-11-09", "vet_id": "V-05",
                                     "vaccines": [{"vaccine": "rabies", "batch_number": "RB-5520"}]}],
            ["record_vaccinations", {"patient_id": "A-115", "date_given": "2026-11-09", "vet_id": "V-05",
                                     "vaccines": [{"vaccine": "lepto", "batch_number": "LP-2093"}]}],
            ["record_vaccinations", {"patient_id": "A-115", "date_given": "2026-11-09", "vet_id": "V-05", "vaccines": [
                {"vaccine": "kennel_cough", "batch_number": "KC-4410", "route": "subcutaneous"}]}],
            ["record_vaccinations", {"patient_id": "A-111", "date_given": "2026-11-09", "vet_id": "V-05", "vaccines": [
                {"vaccine": "felv", "batch_number": "FL-3012"}, {"vaccine": "fvrcp", "batch_number": "FV-1180"}]}],
            ["record_vaccinations", {"patient_id": "A-102", "date_given": "2026-11-09", "vet_id": "V-05",
                                     "vaccines": [{"vaccine": "felv", "batch_number": "FL-3012"}]}]],
    "M06": [["record_weight", {"patient_id": "A-109", "date": "2026-11-09", "weight_grams": 1180, "body_condition": 5}],
            ["reschedule_appointment", {"appointment_id": "AP-3011", "date": "2026-11-21", "time": "10:40",
                                        "vet_id": "V-03"}]],
    "M07": [["allocate_stock", {"order_id": "OR-5150", "picks": [{"lot_id": "LT-1502", "qty": 4},
                                                                {"lot_id": "LT-1201", "qty": 6}]}],
            ["allocate_stock", {"order_id": "OR-5150", "picks": [{"lot_id": "LT-1501", "qty": 4}]}],
            ["allocate_stock", {"order_id": "OR-5150", "picks": [{"lot_id": "LT-1201", "qty": 6}]}],
            ["create_shipment", {"order_id": "OR-5150", "carrier": "parceldirect", "service": "standard",
                                 "ship_date": "2026-11-11", "parcels": [
                                     {"weight_g": 12000, "length_cm": 50, "width_cm": 40, "height_cm": 30}]}]],
    "M09": [["change_carrier", {"order_id": "OR-5163", "carrier": "northline", "service": "standard"}],
            ["create_shipment", {"order_id": "OR-5163", "carrier": "northline", "service": "standard",
                                 "ship_date": "2026-11-10", "parcels": [
                                     {"weight_g": 8800, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}],
            ["change_carrier", {"order_id": "OR-5162", "carrier": "courierblue", "service": "express"}],
            ["create_shipment", {"order_id": "OR-5162", "carrier": "courierblue", "service": "express",
                                 "ship_date": "2026-11-10", "parcels": [
                                     {"weight_g": 6900, "length_cm": 30, "width_cm": 40, "height_cm": 30}]}],
            ["create_shipment", {"order_id": "OR-5161", "carrier": "swiftpost", "service": "standard",
                                 "ship_date": "2026-11-10", "parcels": [
                                     {"weight_g": 5750, "length_cm": 40, "width_cm": 30, "height_cm": 30}]}]],
    "C2": [["move_stock", {"lot_id": "LT-1302", "to_bin": "BN-C02", "qty": 18}],
           ["move_stock", {"lot_id": "LT-1305", "to_bin": "BN-C03", "qty": 25}],
           ["quarantine_lot", {"lot_id": "LT-1301", "reason": "temperature_breach"}]],
    "M14": [["relocate_sample", {"sample_id": "SM-1004", "rack_id": "RK-42", "slot": 2}],
            ["relocate_sample", {"sample_id": "SM-1003", "rack_id": "RK-41", "slot": 6}],
            ["relocate_sample", {"sample_id": "SM-1002", "rack_id": "RK-41", "slot": 5}],
            ["relocate_sample", {"sample_id": "SM-1001", "rack_id": "RK-41", "slot": 3}]],
    "M16": [["destroy_samples", {"sample_ids": ["SM-0303"], "method": "incineration", "witness_id": "PS-05"}],
            ["destroy_samples", {"sample_ids": ["SM-0462", "SM-0461"], "method": "incineration", "witness_id": "PS-03"}],
            ["destroy_samples", {"sample_ids": ["SM-0302"], "method": "autoclave", "witness_id": "PS-05"}],
            ["destroy_samples", {"sample_ids": ["SM-0301"], "method": "autoclave", "witness_id": "PS-05"}]],
    "M17": [["create_aliquots", {"sample_id": "SM-1201", "aliquots": [
        {"volume_ul": 100, "rack_id": "RK-52", "slot": 5}, {"volume_ul": 100, "rack_id": "RK-52", "slot": 3},
        {"volume_ul": 100, "rack_id": "RK-52", "slot": 6}]}],
            ["check_out_sample", {"sample_id": "SM-2000", "to_person": "PS-04", "purpose": "assay",
                                  "due_back": "2026-11-20"}]],
    "M18": [["transfer_custody", {"sample_ids": ["SM-0703"], "from_person": "PS-07", "to_person": "PS-01",
                                  "reason": "handover"}],
            ["transfer_custody", {"sample_ids": ["SM-0705", "SM-0702", "SM-0701"], "from_person": "PS-07",
                                  "to_person": "PS-04", "reason": "handover"}]],
    "M19": [["reschedule_fixture", {"fixture_id": "FX-212", "date": "2026-11-14", "kickoff": "16:00",
                                    "venue_id": "VN-4", "pitch": 4}],
            ["reschedule_fixture", {"fixture_id": "FX-210", "date": "2026-11-14", "kickoff": "13:00",
                                    "venue_id": "VN-4", "pitch": 1}],
            ["reschedule_fixture", {"fixture_id": "FX-211", "date": "2026-11-14", "kickoff": "10:00",
                                    "venue_id": "VN-4", "pitch": 2}]],
    "M21": [["record_result", {"fixture_id": "FX-203", "home_goals": 1, "away_goals": 4,
                               "cards": [{"player_id": "PL-604", "card": "red", "minute": 50}]}],
            ["issue_suspension", {"player_id": "PL-604", "matches": 1, "reason": "red_card"}],
            ["submit_team_sheet", {"fixture_id": "FX-220", "team_id": "TM-22", "captain_id": "PL-601", "players": [
                {"player_id": "PL-608", "shirt": 12, "starting": True}, {"player_id": "PL-609", "shirt": 14,
                                                                         "starting": False},
                {"player_id": "PL-607", "shirt": 7, "starting": True}, {"player_id": "PL-606", "shirt": 6, "starting": True},
                {"player_id": "PL-605", "shirt": 5, "starting": True}, {"player_id": "PL-603", "shirt": 3, "starting": True},
                {"player_id": "PL-602", "shirt": 2, "starting": True}, {"player_id": "PL-601", "shirt": 1,
                                                                        "starting": True}]}]],
    "M26": [["file_condition_report", {"object_id": "OB-1701", "date": "2026-11-13", "condition": "good",
                                       "examiner_id": "SF-02"}],
            ["update_loan", {"loan_id": "LN-410", "remove_objects": ["OB-1703"]}],
            ["update_loan", {"loan_id": "LN-410", "max_lux": 50, "courier": True}],
            ["approve_loan", {"loan_id": "LN-410"}]],
    "M28": [["move_object", {"object_id": "OB-1601", "to_location": "GL-3", "display_until": "2027-03-31"}],
            ["move_object", {"object_id": "OB-1603", "to_location": "GL-1", "display_until": "2026-12-20"}],
            ["move_object", {"object_id": "OB-1602", "to_location": "GL-1", "display_until": "2027-02-28"}]],
    "M30": [["file_condition_report", {"object_id": "OB-1105", "date": "2026-11-13", "condition": "poor",
                                       "examiner_id": "SF-03", "issues": [
                                           {"area": "base of the plinth", "issue": "crack", "severity": "severe"}]}],
            ["cancel_loan", {"loan_id": "LN-420", "reason": "condition_concern"}],
            ["send_to_conservation", {"object_id": "OB-1105", "treatment": "repair", "priority": "urgent",
                                      "estimated_days": 30}],
            ["file_condition_report", {"object_id": "OB-1104", "date": "2026-11-13", "condition": "fair",
                                       "examiner_id": "SF-03", "issues": [
                                           {"area": "lower right corner", "issue": "stain", "severity": "minor"}]}],
            ["file_condition_report", {"object_id": "OB-1101", "date": "2026-11-13", "condition": "good",
                                       "examiner_id": "SF-03"}]],
}


def _verify(expect_ids=None):
    ids = [t["id"] for t in TASKS]
    if expect_ids is not None:
        assert ids == expect_ids, ids
    assert len(set(ids)) == len(ids)
    by_id = dict((t["id"], t) for t in TASKS)
    for ename, cls in ENVS.items():
        assert 10 <= len(cls.TOOLS) <= 14, (ename, len(cls.TOOLS))
        json.dumps(cls.TOOLS)
        names = [tool["function"]["name"] for tool in cls.TOOLS]
        assert len(set(names)) == len(names)
        assert any(n.startswith("get_") and n.endswith(("_info", "_policy", "_rules")) for n in names), (ename, "policy")
        assert any("page" in tool["function"]["parameters"]["properties"] for tool in cls.TOOLS), (ename, "paging")
        assert any(tool["function"]["parameters"]["properties"].get(k, {}).get("type") == "array" and
                   tool["function"]["parameters"]["properties"][k]["items"]["type"] == "object"
                   for tool in cls.TOOLS for k in tool["function"]["parameters"]["properties"]), (ename, "nested")
        for tool in cls.TOOLS:
            fn = tool["function"]
            assert tool["type"] == "function" and fn["parameters"]["type"] == "object"
            assert hasattr(cls, "_t_" + fn["name"]), fn["name"]
            _schema_ok(fn["parameters"], fn["name"])
            for r in fn["parameters"]["required"]:
                assert r in fn["parameters"]["properties"], (fn["name"], r)
    ok = 0
    rng = random.Random(7)
    report = []
    perturbed = [0, 0]
    for t in TASKS:
        tid = t["id"]
        cls = ENVS[t["env"]]
        tools = dict((x["function"]["name"], x["function"]) for x in cls.TOOLS)
        words = len(t["prompt"].split())
        if t["family"] == "single":
            assert 25 <= words <= 150, (tid, "prompt words", words)
            assert len(t["reference"]) == 1 and not _is_read(t["reference"][0][0]), (tid, "single-call reference")
            assert not t["ref_errors"], tid
            assert len(set(t["features"])) >= 2 and set(t["features"]) <= set("TUNDOPX"), (tid, "features")
        else:
            assert 50 <= words <= 200, (tid, "prompt words", words)
            assert 6 <= len(t["reference"]) <= 14, (tid, "reference length", len(t["reference"]))
            assert len(set(t["features"])) >= 3 and set(t["features"]) <= set("abcdefg"), (tid, "features")
            assert _rereads(t) >= 1, (tid, "no re-read of state that a write changed")
        json.dumps(t["init"])
        json.dumps(t["reference"])
        json.dumps(t["trap"])
        _no_empty(t["reference"], tid)
        n_ent = _prompt_entities(t)
        # (a) untouched initial state must fail
        env = cls(t["init"])
        assert env.state == t["init"] and env.state is not t["init"]
        assert t["check"](env.state) is False, (tid, "check passes on the initial state")
        # (b) reference solves it; errors only where declared; reads never change the state
        chars = 0
        for n, (name, args) in enumerate(t["reference"]):
            assert name in tools, (tid, "unknown tool in reference", name)
            before = copy.deepcopy(env.state)
            res = env.call(name, copy.deepcopy(args))
            chars += len(json.dumps(res))
            assert _is_err(res) == (n in t["ref_errors"]), (tid, "reference step", n, name, res)
            if _is_read(name) or _is_err(res):
                assert env.state == before, (tid, "state changed by read/error step", n)
            else:
                assert env.state != before, (tid, "reference write changed nothing", n, name)
        assert t["check"](env.state) is True, (tid, "reference does not satisfy check")
        assert chars < 45000, (tid, "reference transcript too long for the context window", chars)
        json.dumps(env.state)
        solved = _clone(env.state)
        # every read tool that takes no required arguments must leave a solved state solved
        for fn in tools.values():
            if not fn["parameters"]["required"] and _is_read(fn["name"]):
                env.call(fn["name"], {})
        assert t["check"](env.state) is True, (tid, "read tools changed the state")
        # (c) the trap must change something and fail, and replayed on the solved state it must fail too
        env = cls(t["init"])
        for name, args in t["trap"]:
            assert name in tools, (tid, "unknown tool in trap", name)
            json.dumps(env.call(name, copy.deepcopy(args)))
        assert env.state != t["init"], (tid, "trap changed nothing")
        assert t["check"](env.state) is False, (tid, "trap passes the check")
        env = cls(solved)
        for name, args in t["trap"]:
            env.call(name, copy.deepcopy(args))
        assert env.state == solved or t["check"](env.state) is False, (tid, "solved state + trap passes the check")
        # (d) dropping any single successful write from the reference must fail
        writes = [n for n, (name, _) in enumerate(t["reference"]) if not _is_read(name) and n not in t["ref_errors"]]
        for skip in writes:
            env = cls(t["init"])
            for n, (name, args) in enumerate(t["reference"]):
                if n != skip:
                    env.call(name, copy.deepcopy(args))
            assert t["check"](env.state) is False, (tid, "check passes although reference step %d was skipped" % skip)
        # (e) alternative valid write orders must pass
        valid, nw = _alt_orders(t, rng)
        # (f) single-call: every perturbed argument that changes the outcome must fail
        if t["family"] == "single":
            a, b = _perturb_single(t)
            perturbed[0] += a
            perturbed[1] += b
        report.append((tid, len(t["reference"]), nw, valid, words, n_ent, chars, t["family"]))
        # init must not have been mutated by any of the above
        assert t["check"](copy.deepcopy(t["init"])) is False
        ok += 1
    # (g) hand-written alternative routes must pass as well
    for key, route in sorted(_ALT_ROUTES.items()):
        t = by_id[key.rstrip("b")]
        assert route != t["reference"], (key, "alternative route is the reference itself")
        env = ENVS[t["env"]](t["init"])
        for name, args in route:
            env.call(name, copy.deepcopy(args))   # may include a refused attempt, as a model's route would
        assert t["check"](env.state) is True, (key, "alternative route fails check")
    calls = _fuzz()
    for t in TASKS:   # fuzz must not have leaked into the fixtures
        env = ENVS[t["env"]](t["init"])
        for name, args in t["reference"]:
            env.call(name, copy.deepcopy(args))
        assert t["check"](env.state) is True, (t["id"], "fixture mutated by fuzz")
    return ok, calls, report, perturbed


if __name__ == "__main__":
    _expect = ["S%02d" % n for n in range(1, 31)] + ["M%02d" % n for n in range(1, 31)] + ["C%d" % n for n in range(1, 9)]
    _ok, _calls, _report, _pert = _verify(_expect)
    _env_order = ["vet", "warehouse", "lab", "league", "museum"]
    assert sorted(_env_order) == sorted(ENVS)
    for _n, _e in enumerate(_env_order):     # 6 single + 6 multi per environment, in blocks of six
        assert [t["env"] for t in TASKS[6 * _n: 6 * _n + 6]] == [_e] * 6, _e
        assert [t["env"] for t in TASKS[30 + 6 * _n: 36 + 6 * _n]] == [_e] * 6, _e
        assert len([t for t in TASKS if t["env"] == _e and t["family"] == "single" and not t["calibration"]]) == 6, _e
        assert len([t for t in TASKS if t["env"] == _e and t["family"] == "multi" and not t["calibration"]]) == 6, _e
        assert len([t for t in TASKS if t["env"] == _e and t["calibration"]]) >= 1, _e
    _cal = [t for t in TASKS if t["calibration"]]
    assert len([t for t in _cal if t["family"] == "single"]) == 4 and len([t for t in _cal if t["family"] == "multi"]) == 4
    for _t in TASKS:   # sizes of the initial states
        _n = sum(len(v) for v in _t["init"].values() if _is_collection(v))
        assert 20 <= _n, (_t["id"], _n)
    _multi = [t for t in TASKS if t["family"] == "multi"]
    print("envs: " + ", ".join("%s (%d tools)" % (k, len(ENVS[k].TOOLS)) for k in _env_order))
    for _fam, _label in (("single", "single-call"), ("multi", "multi-step")):
        _r = [r for r in _report if r[7] == _fam]
        _rt = [r for r in _r if not r[0].startswith("C")]
        print("%s family: %d test + %d calibration tasks; reference calls min %d, max %d (test set %d-%d); "
              "prompt words min %d, max %d" % (_label, len(_rt), len(_r) - len(_rt), min(r[1] for r in _r),
                                               max(r[1] for r in _r), min(r[1] for r in _rt), max(r[1] for r in _rt),
                                               min(r[4] for r in _r), max(r[4] for r in _r)))
    print("multi-step: %d of %d tasks have a step the policy refuses (%d refused steps in all); every one re-reads "
          "state a write changed" % (len([t for t in _multi if t["ref_errors"]]), len(_multi),
                                     sum(len(t["ref_errors"]) for t in _multi)))
    print("single-call perturbations: %d perturbed calls, %d of them succeeded with a different outcome, all fail check"
          % (_pert[0], _pert[1]))
    print("alternative write orders: %d valid orders replayed over %d tasks, all pass check" % (
        sum(r[3] for r in _report), len(_report)))
    print("alternative routes (different calls, same outcome): %d routes, all pass check" % len(_ALT_ROUTES))
    print("fuzz: %d calls, 0 exceptions, state unchanged on every error" % _calls)
    print("OK %d/%d" % (_ok, len(TASKS)))
