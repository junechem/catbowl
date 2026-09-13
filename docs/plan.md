# Plan

Where the rig is, and what happens next. Updated 2026-09-13.

## Where it is now

The hardware works. One bowl is live (`bowl1`): a cardboard disc lying on the
bowl, turned aside by a single servo on PCA9685 channel 1, now glued to the lid
rather than taped. The camera is a Logitech Brio 100 on USB. The status page
serves the live view, the sorting queue and the browser at
`http://rjwpi.local:8080/`.

The service runs `--no-model`, which is the important caveat: **there is no
trained classifier yet.** In that mode anything the cat detector finds opens the
lid - any cat, a hand, a passing dog. What exists is the training set, banked by
the rig itself and sorted by hand through `/sort`:

| bucket | photos |
| --- | --- |
| J | 666 |
| K | 579 |
| F | 551 |
| M (more than one cat) | 7 |
| discard | 1059 |
| unsorted | 702 |

Everything lives on the Pi under `data/collected/`, which is gitignored: the
photos exist in exactly one place and no git operation can touch them.

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

## Next steps, in order

1. **Train the first classifier** on the 1796 hand-sorted photos.
   `catbowl train --data data/collected --labels J K F`. The `--labels` flag is
   what keeps `discard`, `unsorted` and `proposed` from becoming classes of
   their own. Note the suggested threshold it prints.
2. **Presort the 702 unsorted photos with it.** `catbowl presort` files copies
   into `data/collected/proposed/<label>/`, never into the hand-sorted buckets,
   with anything under the threshold going to `proposed/unsure/`. This doubles
   as the first real test of the model: a human checks the proposals, and how
   many need correcting is the score.
3. **Move the verified proposals** into `data/collected/J|K|F/` and **retrain**
   on the larger set.
4. **Switch the bowl over to recognition.** Drop `--no-model` from the service so
   a lid only lifts for a confirmed cat. This is the point of the whole rig, and
   it is the step that needs a model good enough to trust.
5. **Rename the cats.** The config still says mochi/pepper/biscuit while the
   photos say J/K/F. The `cat:` field of each bowl has to match the classifier's
   labels exactly, so this has to happen together with step 4.

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
