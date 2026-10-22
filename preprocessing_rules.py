"""Per-column cleaning rules for the BRFSS data.

The tables are plain data, so the preprocessing can switch each
group on or off. "codebook" marks rules checked in the official BRFSS 2015
codebook, "guess" marks rules inferred from value patterns in the data.
Running this file prints how many rows each rule touches.
"""

import numpy as np

from data_loading import get_train_data

# Identifier-like or date-like columns (guess, from value patterns).
DROP_ID_DATE = ["SEQNO", "_PSU", "IDATE", "IDAY", "HIVTSTD3"]

# Raw columns that have a cleaner derived twin (checked with NaN shares).
DROP_RAW_WITH_TWIN = [
    "ALCDAY5",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",
    "STRENGTH",
    "WEIGHT2",
    "HEIGHT3",
]

# Numbers that are category labels or unit codes, so a numeric scale is wrong.
DROP_CATEGORY_OR_UNIT = ["EXRACT11", "EXRACT21", "BLDSUGAR", "FEETCHK2", "LONGWTCH"]

# Rule format: "nan" codes become NaN, "zero" codes become 0, "set" maps a
# code to another value.
COUNT_RULE = {"nan": [77, 99], "zero": [88]}
COLUMN_RULES = {}

# Count columns: 88 = none, 77 = don't know, 99 = refused (codebook).
# The seven AD* columns after ADPLEASR follow the same pattern (guess).
for _name in [
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "DRNK3GE5",
    "DOCTDIAB",
    "FEETCHK",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
]:
    COLUMN_RULES[_name] = COUNT_RULE

COLUMN_RULES.update(
    {
        "CHILDREN": {"nan": [99], "zero": [88]},
        "HHADULT": {"nan": [77, 99]},
        "AVEDRNK2": {"nan": [77, 99]},
        "MAXDRNKS": {"nan": [77, 99]},
        "DIABAGE2": {"nan": [98, 99]},
        # 98 = never heard of the A1C test; mapping it to 0 checks is a guess.
        "CHKHEMO3": {"nan": [77, 99], "zero": [88, 98]},
        # 97 = age 10 or younger; mapping it to 10 is a guess.
        "ASTHMAGE": {"nan": [98, 99], "set": {97: 10}},
        "ASERVIST": {"nan": [98], "zero": [88]},
        "ASDRVIST": {"nan": [98], "zero": [88]},
        "ASRCHKUP": {"nan": [98, 99], "zero": [88]},
        "ASACTLIM": {"nan": [777, 999], "zero": [888]},
        # Hours worked: 97 = don't know, 98 = zero hours, 99 = refused.
        "SCNTWRK1": {"nan": [97, 99], "zero": [98]},
        "SCNTLWK1": {"nan": [97, 99], "zero": [98]},
        # Derived columns with sentinel codes (codebook).
        "DROCDY3_": {"nan": [900]},
        "_DRNKWEK": {"nan": [99900]},
        "MAXVO2_": {"nan": [999]},
        # Same 3,943 rows as MAXVO2_ (guess).
        "FC60_": {"nan": [999]},
        "_AGEG5YR": {"nan": [14]},
        "_CHISPNC": {"nan": [9]},
        "_PRACE1": {"nan": [77, 99]},
        "_MRACE1": {"nan": [77, 99]},
        # Same pattern as _MRACE1 (guess).
        "_CRACE1": {"nan": [77, 99]},
        "_CPRACE": {"nan": [77, 99]},
    }
)


def rule_impact(col, rule):
    """Return how many rows each code in a rule affects in a column."""
    codes = list(rule.get("nan", [])) + list(rule.get("zero", []))
    codes += list(rule.get("set", {}))
    return {code: int(np.sum(col == code)) for code in codes}


def main():
    x, y, ids, names = get_train_data()
    names = [str(s) for s in names]
    dropped = DROP_ID_DATE + DROP_RAW_WITH_TWIN + DROP_CATEGORY_OR_UNIT
    unknown = [n for n in dropped + list(COLUMN_RULES) if n not in names]
    print("Names not found in the data: %s" % unknown)
    print("Dropped by default: %d columns" % len(set(dropped)))
    for name, rule in COLUMN_RULES.items():
        if name in names:
            counts = rule_impact(x[:, names.index(name)], rule)
            print("%-9s %s" % (name, counts))


if __name__ == "__main__":
    main()
