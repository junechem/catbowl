"""Per-cat feeding station: recognise the cat at each bowl, lift only its lid."""

__version__ = "0.1.0"

UNKNOWN = "unknown"
# More than one cat in front of the bowl. Reported instead of a cat's name, so
# it can never win a vote for the owner, and an open lid treats it as an
# intruder and comes down.
CROWD = "_crowd"
