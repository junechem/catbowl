# Plan

Where the rig is, and what happens next. Updated 2026-09-13.

## Where it is now

The hardware works. One bowl is live (`bowl1`): a cardboard disc lying on the
bowl, turned aside by a single servo on PCA9685 channel 1, now glued to the lid
rather than taped. The camera is a Logitech Brio 100 on USB. The status page
serves the live view, the sorting queue and the browser at
`http://rjwpi.local:8080/`.

A classifier now exists, trained on the Pi from the photos the rig banked and a
human sorted: **J 925, K 722, F 710, and 1232 discards**. The discards are
learnt as a negative class (`_other`, "none of the cats") rather than as a
fourth cat, so a lid can never open for one.

    raw accuracy 90.4%   |   one cat called another: 23 of 3589 (0.6%)

The overall figure is lower than the cats-only model's 96.5%, and better: almost
all of the new error is a cat confused with junk, the least harmful mistake
available, while the error that actually matters - the wrong cat fed - fell from
1.7% to 0.6%. Threshold `min_confidence: 0.85`.

Everything lives on the Pi under `data/collected/`, which is gitignored: the
photos exist in exactly one place and no git operation can touch them. The rig
files each new photo it takes into `proposed/<its guess>`, so the review queue
builds itself and every day's photos are a fresh test of the model.

## How a lid decides to open

Opening asks for **one confident sighting** (`policy.open_votes: 1`), not a
consensus. A cat walking up has no history to consult, and every frame it waits
is a frame it spends at a shut bowl. The protection a vote window offers here is
thinner than it looks: consecutive frames of the same cat in the same light are
not independent samples, so a hard frame errs the same way several times over
and the window agrees with itself. A frame the model is genuinely unsure of now
lands in `_other`, which can never win a vote at all.

Everything after the open - intruder, crowd, close - still needs
`votes_required` of the last `vote_window` frames, because by then there is
context to consult. A wrong open costs the wrong cat a mouthful before the
intruder rule shuts the lid; a slow open costs the right cat its meal every
time.

Two guards remain on the open: a cat cannot open the lid while another cat is
better represented in the window, and a sighting that has scrolled out of the
window is not a sighting.

## The two safety questions, answered

**Does it refuse to open when it cannot tell which cat it is?** Yes, by two
independent gates. Any prediction below `recognition.min_confidence` is reported
as `unknown`, and `unknown` can never win a vote (`VoteTracker.decision`). Above
it, a label still has to hold `votes_required` of the last `vote_window` frames
before a lid moves. A cat the model has never seen produces a scatter of
low-confidence guesses, which is a decision of "none".

**Does it know when there is more than one cat?** It does now. This was a real
hole: the classifier is only ever asked "which cat is this crop", and with two
cats at one bowl the honest answer about the crop is still the wrong answer
about the bowl - the other cat is standing right there and eats through the open
lid. The detector is the only part that can see both, so it now counts the cats
it confirms and carries the count on the detection. More than one, and the frame
is reported as `_crowd` instead of a name: it cannot open a lid, and it closes
an open one through the existing intruder path.

Two limits worth knowing. The count comes from `ssdlite`, so it needs
`detector.type: hybrid` (the default) - plain motion detection cannot count. And
between confirmations (`confirm_every_s`, 2 s) the last count stands, so a
second cat arriving is noticed within about two seconds rather than instantly.

## How the one bowl serves three cats

There is one bowl and there are three cats, and there will be one bowl until
there is time to build more. So the bowl feeds all three, on different terms:

| cat | rule |
| --- | --- |
| K | opens whenever she walks up, for as long as she stays |
| J | two minutes of open lid per rolling hour |
| F | two minutes of open lid per rolling hour |

The allowance is a **budget, not one meal an hour**. A cat startled away after
ten seconds keeps the rest of its two minutes and can come back for it. Time is
charged while the lid is open - including the `close_delay_s` it stays up behind
a cat that has already wandered off - and charged as the meal happens, so a cat
that never leaves is still billed. An hour after each mouthful, that mouthful's
worth of allowance comes back.

Running out closes the lid mid-meal (`reason: ration`) and refuses the next
approach with a `denied` event saying how long until it can eat again. A cat
arriving while another is eating still ends the meal - it gets its own turn on
its own allowance, rather than sharing the open lid.

Allowances live in memory. A restart hands every cat a full two minutes again,
which errs towards feeding a cat twice rather than starving one that has eaten
nothing.

## Next steps, in order

1. **Deploy and switch to recognition.** Drop `--no-model` from the service so a
   lid only lifts for a confirmed cat, with `min_confidence: 0.85`.
2. **Watch the first day.** The numbers that matter are wrong opens (the wrong
   cat fed) and refusals of the right cat. `/` shows each cat's remaining
   allowance; the event log shows every open, close and denial with its reason.
3. **Keep checking the proposals.** The running rig now files each photo it
   takes into `proposed/<its guess>`, so the review queue builds itself and
   every session's worth of photos is a fresh test of the model.
4. **Retrain** as the checked piles grow, and again whenever the cats' coats
   change with the seasons.
5. **Build bowls 2 and 3**, and give J and F their own, at which point their
   rations can go.

## Time, and what uses it

Every listing on `/browse` is already in time order - the capture filename
carries the timestamp, so a visit's frames sit together in the grid whichever
folder they are in.

`presort` uses time as evidence, as above, and the lid's rules use a second of
it (see *How a lid decides to open*). Beyond that the rig carries no memory of who was just here. A visit-level prior
with hysteresis - harder to switch identity mid-meal than to keep it - is the
obvious next refinement once the model is trusted.

## Open questions

- **Crop quality.** `HybridCatDetector` was fixed to crop the whole animal
  rather than the tuft of fur that moved, but photos captured before that fix
  are still in the pile. If accuracy is poor, bad crops in the training set are
  the first place to look.
- **Second and third bowls.** `bowl2` and `bowl3` are configured and disabled.
  Each needs a camera, a servo and a lid.
- **Camera dropouts.** The servo moving has knocked the camera off USB (kernel
  `error -71`). Hardware first - cable routing, a ferrite, a capacitor across
  the servo supply - but the software should also reopen a camera that has
  dropped rather than blinding itself until the next restart.
