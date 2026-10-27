# SpExplAIn tutorial: explaining the predictions of a neural network

This hands-on tutorial shows how to explain the predictions of a neural network with SpExplAIn,
starting only from the network, the data, and the original units of the measurements.
You type every command yourself, look at the results, and then try the same with other inputs.

* **Part 1, first steps,** needs no prior knowledge:
  a network that predicts the risk of a heart attack from the measurements of a patient,
  a handful of patients, and default settings everywhere.
* **Part 2, an experimental playground for domain and verification experts,** explores further:
  other ways of computing explanations, other networks, and other data (loans, handwritten digits).

All commands run in seconds unless stated otherwise.

## Before you start

The tutorial is designed for the Docker image of SpExplAIn (see [`docker/tacas27`](../docker/tacas27)),
but it works equally when run directly on your computer,
once the dependencies are installed and the tool is built
(see [Building from source](../README.md#building-from-source)).
All outputs (explanations, pictures, and formulas) are written into the directory `$OUTPUT/tutorial`.

**In Docker:** run the image with a directory `<dir>` of your computer for the outputs:
```
OUTPUT=<dir> docker/run.sh docker/tacas27
```
Optionally, add `X11=1` to connect the container to the display of your computer
(see [Showing the pictures in windows](../docker/tacas27/README.md#optional-showing-the-pictures-in-windows));
on Linux, a container started with `X11=1` cannot be restarted later (by `docker start`), start a new one instead.
Inside the container, you are in the root of the repository (`~/spexplain`),
and the variable `OUTPUT` is already set to `~/spexplain/data/output`,
which is the directory `<dir>` of your computer.

**On your computer:** run the commands from the root of the repository.
Part 2 also uses the verifier Marabou, built by `make marabou`.
Choose a directory for the outputs:
```
OUTPUT=output
```

The tutorial writes into its own directory in `$OUTPUT`:
```
OUT=$OUTPUT/tutorial
mkdir -p $OUT
```

The commands that create a picture or a formula open it right away in a window (`--show`);
close the window of a picture to continue.
**Without a display** (e.g., in Docker without `X11=1`), leave out `--show`
and open the files in `$OUT` instead (in Docker, in the directory `<dir>/tutorial` of your computer).
The files are written in both cases.

To keep the commands short, set the three inputs once:
```
MODEL=data/models/heart_attack/heart_attack_50x1.nnet
DATASET=data/datasets/heart_attack/heart_attack_s100_scaled_short.csv
DOMAINS=data/domains/heart_attack/heart_attack.csv
```
* `MODEL` is the neural network that predicts the risk of a heart attack.
  In part 1, it is a black box: we only see what it predicts, not how.
* `DATASET` contains 8 patients: their measurements (age, blood pressure, …) and the recorded risk.
  The measurements are stored rescaled to the range from 0 to 1, as the network expects them.
* `DOMAINS` gives the original units of the measurements (e.g., the age from 29 to 77 years),
  so that we can read everything in the usual units.

## Part 1: First steps: explaining a prediction

### Step 1: What is measured and what is predicted

```
./spexplain list vars $DATASET --domains $DOMAINS
```
This lists the measurements (the *features*) with their ranges in the original units.
Each measurement has a short name `x1`, `x2`, …, which we use later to refer to it:
`x1` is the age, `x4` the blood pressure, `x5` the cholesterol, `x8` the maximum heart rate, and so on.

```
./spexplain list classes $DATASET
```
This lists what the network predicts (the *classes*): `High risk` (0) or `Low risk` (1).

### Step 2: The patients and the predictions

```
./spexplain list samples $DATASET --model $MODEL --domains $DOMAINS
```
Each row is one patient (a *sample*), numbered from 1, with all measurements in the original units
(the message above the table reminds you of that).
The column `Expected` is the recorded risk and `Computed` is the risk predicted by the network.
A row marked by `!` is a patient where the network disagrees with the record: here, patient 8.

We will look at two patients: patient 2 (predicted `Low risk`, as recorded)
and patient 8 (predicted `High risk`, but recorded as `Low risk`).

### Step 3: Compute the explanations

```
./spexplain $MODEL $DATASET -e $OUT/phi.txt
```
For each patient, this computes an *explanation* of the prediction:
a region of measurements around the patient where the network is **guaranteed**
to give the same prediction as for the patient.
In other words, changing the measurements of the patient within this region cannot change the prediction.

The explanations are written into `$OUT/phi.txt`, one line per patient.

### Step 4 (optional): An explanation as a formula is not readable

```
./spexplain interpret $OUT/phi.txt mathexpr $DATASET -i 2 --visualization-domains $DOMAINS --output-visualization $OUT/phi.pdf --show
```
This writes the explanation of patient 2 (`-i 2`) as a mathematical formula in the original units
into `$OUT/phi_s2.pdf` (`_s2` stands for the patient).
Even rendered in this user-friendly way, the formula is unreadable due to its complexity:
it consists of more than twenty conditions, each combining almost all measurements at once.
Hence, the explanation requires an interpretation, which the following steps provide.

### Step 5: Look at two measurements at a time

A good way to understand an explanation is to look at two measurements at a time,
keeping all the others as they are for the patient.
This is only a limitation of the visualization, so that the pictures are readable for humans
(three measurements at a time would also be possible, but it is not implemented):
the computed explanation itself covers all measurements at once.
Choose the two measurements from the list of step 1, e.g., the age (`x1`) and the maximum heart rate (`x8`):
```
./spexplain interpret $OUT/phi.txt 'slice vars x1 x8' --visualize $DATASET -i 2 --model $MODEL --visualization-domains $DOMAINS --output-visualization $OUT/slice_x1_x8.png --show
```
The picture `$OUT/slice_x1_x8_s2.png` shows:
* the **patient** as the black diamond;
* the **explanation** as the red area:
  all combinations of the age and the maximum heart rate for which the prediction is guaranteed to stay
  `Low risk`;
* the **prediction of the network** in the background colors (thanks to `--model`):
  in the bottom part of the picture, the network would predict `High risk`:
  below a certain maximum heart rate, which grows with the age
  (from about 105 at the age of 30 to about 140 at the age of 77);
  the explanation reaches right up to this border, but never crosses it;
  it does not cover the whole blue area, because it is a guarantee that is simple to describe,
  not the largest possible area;
* the **caption** with the number of the patient and the predicted class.

Now look at patient 8 (`-i 8`) and another pair, the blood pressure (`x4`) and the maximum heart rate (`x8`):
```
./spexplain interpret $OUT/phi.txt 'slice vars x4 x8' --visualize $DATASET -i 8 --model $MODEL --visualization-domains $DOMAINS --output-visualization $OUT/slice_x4_x8.png --show
```
For patient 8, the caption of `$OUT/slice_x4_x8_s8.png` says `High risk (expected: Low risk)`,
because the network disagrees with the record.
At a low blood pressure and a high maximum heart rate, in the top left corner, the background turns blue:
there, the network would predict `Low risk`.
The explanation stops right before that area, as it must.

### Step 6: The largest safe change

Within the explanation, which change of the two measurements is the largest?
```
./spexplain interpret $OUT/phi.txt 'diam vars x1 x8' --visualize $DATASET -i 2 --model $MODEL --visualization-domains $DOMAINS --output-visualization $OUT/diam.png --show
```
The dashed line in `$OUT/diam_s2.png` is the *diameter* of the explanation:
the longest segment within the red area.
This simple object can also be written as a short formula:
```
./spexplain interpret $OUT/phi.txt 'diam vars x1 x8' --visualize mathexpr $DATASET -i 2 --visualization-domains $DOMAINS --output-visualization $OUT/diam.pdf --show
```
The formula in `$OUT/diam_s2.pdf` bounds the age from below
and expresses the maximum heart rate as a growing function of the age.
In words: *the prediction stays `Low risk` for any age from a certain age on,
if the maximum heart rate grows together with the age along this line.*
Unlike the formula of step 4, this one is readable.

Try the same for patient 8 (`-i 8`) and for other pairs of measurements.

## Part 2: Experimental playground for domain and verification experts

From now on, choose the patients and the measurements yourself:
`./spexplain list vars` shows the measurements, and `./spexplain list samples` the patients,
also only some of them, e.g., `./spexplain list samples 2 8 $DATASET --model $MODEL --domains $DOMAINS --vars x1,x4,x8`.
Each command of part 1 works with any patient (`-i`) and any pair of measurements (`vars`).

### Other ways of computing explanations

The explanations of part 1 were computed with the default settings.
Other settings give different explanations of the same predictions.
Compute some of them into other files:
```
./spexplain $MODEL $DATASET 'itp afactor 0.5, bstrong' -e $OUT/phi_afactor.txt
./spexplain $MODEL $DATASET abductive -V marabou -e $OUT/phi_abductive.txt
./spexplain $MODEL $DATASET trial -V marabou -e $OUT/phi_trial.txt
```
* `itp afactor 0.5, bstrong` is another variant of the default method (interpolation).
* `abductive` with the verifier Marabou (`-V marabou`) keeps only some measurements fixed to the values
  of the patient and lets all the others change freely.
* `trial` with the verifier Marabou is similar, but instead of letting a measurement change freely,
  it allows it to change within a range around the value of the patient.

Each of them is interpreted exactly as in part 1 (steps 4–6), e.g., for patient 8:
```
./spexplain interpret $OUT/phi_abductive.txt mathexpr $DATASET -i 8 --visualization-domains $DOMAINS --output-visualization $OUT/phi_abductive.pdf --show
./spexplain interpret $OUT/phi_trial.txt mathexpr $DATASET -i 8 --visualization-domains $DOMAINS --output-visualization $OUT/phi_trial.pdf --show
```
These formulas are short and readable without further interpretation:
* `abductive` fixes the sex, the chest pain, the maximum heart rate, and the number of major vessels,
  so *whatever the other nine measurements are, the prediction stays `High risk`*;
* `trial` gives ranges instead of fixed values, e.g.,
  *the prediction stays `High risk` as long as the maximum heart rate does not exceed that of the patient*.

Recall that patient 8 is the one where the network disagrees with the record:
the explanations explain the prediction of the network, not the record.

Several explanations can be compared in one picture, each labelled (`--labels`):
```
./spexplain interpret $OUT/phi.txt $OUT/phi_afactor.txt $OUT/phi_abductive.txt $OUT/phi_trial.txt 'slice vars x1 x8' --visualize $DATASET -i 8 --model $MODEL --visualization-domains $DOMAINS --labels itp,afactor,abductive,trial --output-visualization $OUT/compare.png --show
```
Here, the abductive explanation is a line: the maximum heart rate is fixed to that of the patient,
but any age is allowed;
the explanation of `trial` is the whole area below this line: any age, and any maximum heart rate up to that of
the patient.

Methods can also be chained, each improving the result of the previous one.
For instance, `ucore` removes the conditions of an explanation that are not needed.
The option `-s` writes statistics of each patient, including the number of conditions (`#terms`):
```
./spexplain $MODEL $DATASET -e $OUT/phi.txt -s $OUT/stats.txt
./spexplain $MODEL $DATASET 'itp; ucore' -e $OUT/phi_ucore.txt -s $OUT/stats_ucore.txt
grep '#terms' $OUT/stats.txt $OUT/stats_ucore.txt
```
The plain `ucore` is fast, but it removes only what it finds quickly:
here, only one of the 22 conditions of patient 5.
`ucore min` removes as many conditions as possible, but it takes much longer:
from about 10 seconds to more than 20 minutes per patient.
The option `-t` limits the time per patient (in milliseconds);
with half a minute, the following takes about 4 minutes:
```
./spexplain $MODEL $DATASET 'itp; ucore min' -t 30000 -e $OUT/phi_ucore_min.txt -s $OUT/stats_ucore_min.txt
grep '#terms' $OUT/stats_ucore_min.txt
```
The patients that run out of time have no explanation (`<null>` in the file).
The others have about half of the conditions, e.g., patient 5 only 11 of 22:
a simpler explanation, which also holds for more patients.
This does not necessarily show in the pictures of steps 5 and 6, though:
a picture varies only two features (e.g., `x1` and `x8`, the age and the maximum heart rate)
and keeps all the other features at the values of the patient,
whereas the removed conditions often matter only when several features change at once.

### Using what the network does inside

The explanations can also take into account what happens inside the network for the patient
(which of its *neurons* are active).
Fixing the behavior of all neurons (*fix-all*), or only of those that cannot be easily changed (*hybrid*):
```
./spexplain $MODEL $DATASET --fix-default-sample-neuron-activations all -e $OUT/phi_fixall.txt
./spexplain $MODEL $DATASET --fix-default-sample-neuron-activations adversarial -e $OUT/phi_hybrid.txt
./spexplain interpret $OUT/phi.txt $OUT/phi_fixall.txt $OUT/phi_hybrid.txt 'slice vars x1 x8' --visualize $DATASET -i 2 --model $MODEL --visualization-domains $DOMAINS --labels itp,fix-all,hybrid --output-visualization $OUT/compare_neurons.png --show
```
These explanations have a different shape: for patient 2, they are narrower than the default one,
and the explanation of hybrid is larger than that of fix-all.
Interpret them further as in part 1, e.g., by their diameters.

### Other networks

The network of part 1 is small. We can now look inside it:
```
./spexplain list layers $MODEL
```
It has one hidden layer of 50 neurons.
Another network for the same data, with two hidden layers:
```
MODEL2=data/models/heart_attack/heart_attack_50x2.nnet
./spexplain list layers $MODEL2
./spexplain list samples $DATASET --model $MODEL2 --domains $DOMAINS
```
It disagrees with the records for other patients than the first network.
Its explanations are computed and interpreted in the same way, with the default settings:
```
./spexplain $MODEL2 $DATASET -e $OUT/phi_50x2.txt
./spexplain interpret $OUT/phi.txt $OUT/phi_50x2.txt 'slice vars x1 x8' --visualize $DATASET -i 6 --model $MODEL2 --visualization-domains $DOMAINS --labels 50x1,50x2 --output-visualization $OUT/compare_networks.png --show
```
For patient 6, both networks predict `Low risk`, and the background now shows the predictions of the second network.
The explanation of the first network reaches into the area where the second network predicts `High risk`:
an explanation is a guarantee only for the network it was computed for.
The explanation of the second network is smaller.
Other networks for this data are in `data/models/heart_attack`, named by the number of the neurons per layer
and the number of the layers (e.g., `heart_attack_50x4.nnet`).

### Loans

The same steps work for other data, e.g., the approval of loans:
```
MODEL=data/models/loan/loan_50x1.nnet
DATASET=data/datasets/loan/loan_s100_scaled_short.csv
DOMAINS=data/domains/loan/loan.csv
./spexplain list vars $DATASET --domains $DOMAINS
./spexplain list classes $DATASET --model $MODEL
./spexplain list samples $DATASET --model $MODEL --domains $DOMAINS --vars x1-x8
./spexplain $MODEL $DATASET -e $OUT/loan_phi.txt
./spexplain interpret $OUT/loan_phi.txt 'slice vars x5 x8' --visualize $DATASET -i 1 --model $MODEL --visualization-domains $DOMAINS --output-visualization $OUT/loan_slice.png --show
```
Patient 1 is now an applicant whose loan the network approves, although it was rejected.
In the picture of the interest rate (`x5`) and the credit score (`x8`),
the background shows where the network would reject the loan,
and the explanation stays on the side of the approval.
All the settings of this part, including `--fix-default-sample-neuron-activations adversarial`,
work for this network as well.

### Handwritten digits

For images, each pixel is a measurement, so there are too many of them to look at two at a time:
```
MODEL=data/models/mnist/mnist_50x1.nnet
DATASET=data/datasets/mnist/mnist_s100_scaled_short.csv
./spexplain list samples $DATASET --model $MODEL
./spexplain $MODEL $DATASET -e $OUT/mnist_phi.txt
```
`list samples` shows only the first 20 of the 784 pixels (here, a blank corner of the images).
Instead, an explanation can be shown as a picture of a single image within it:
```
./spexplain interpret $OUT/mnist_phi.txt norm --visualize $DATASET -i 1 --output-visualization $OUT/mnist.png --show
```
This takes about 15 seconds.
The picture `$OUT/mnist_s1.png` shows the image of the digit next to the darkest image within its explanation:
an image with much less ink that the network still recognizes as the same digit.
The brightest image is computed by `norm inv`, and both of them by `norm min, inv`
(with `--visualize 'pixel center'`, the digit is shown between them); these take a few minutes.

## Going further

All actions and options are described in the main [README](../README.md),
and each action prints its options with `--help`, e.g., `./spexplain interpret --help` or `./spexplain list --help`.
