# Casein molecule simulation

This repository contains our 8-week (7.5 ECTS) out-of-course scope project simulating the coagulation of casein micelles in skim milk, using Metropolis Monte Carlo.

By Andreas Leerbeck, Mads Daugaard \& Emil Riis-Jacobsen.

Faculty of Science University of Copenhagen.

Handed in: January 21. 2024.


![Simulation example](Data/DisplayImages/simExampleMolecules.png)

# Requirements

The required python 3.10.13 packages can be found in "req.txt", and easily be instaled with anaconda using this command:
```
conda create --name <env> --file req.txt
```

# File overview

##  Important python files

* /Code/forces.py                     - Everything force related
* /Code/molecules.py                  - Molecules and simulation universe
* /Code/img3dto2d.py                  - Convert a simulation to 2D images
* /Code/stats.py                      - Various stat functions for universes
* /Code/video_format.py               - Code for visualizing and processing the cheese video

## Important notebooks 

* runSimulation.ipynb                 - Run a customized simulation
* visSimulation.ipynb                 - Produce visualizations and stats for a given run
* optimalPostProcessing.ipynb         - Experiment with postprocessing (eg. smoothing/thresholds)
* simCombination.ipynb                - Produce visualizations and RSM for a collection of runs


## Important simulation data

* /Data/6P                            - Contains the 10 runs for 6% volume fraction (and the notebook showcasing the results)
* /Data/13P                           - Contains the 10 runs for 13% volume fraction (and the notebook showcasing the results)
* /Data/OtherRuns                     - Contains all other runs for eg. different stepsizes

## Prototypes, tests, and old files

* /Prototypes/Ploting                 - Notebooks for producing plots, some for the report
* /Prototypes/Simulator               - Notebooks showcasing various versions of the simulator
* /Prototypes/TestFiles               - Files for testing functionality and various libraries
* /Prototypes/VisualizationAttempts   - Notebooks for different 3D visualization attempts

