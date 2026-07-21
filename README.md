# FUS Simulator — Open-source Interactive Planning Tool for In Vivo Mouse Transcranial Focused Ultrasound Experiments 
# Copyright (C) 2026 Morse Lab
# Licensed under the GNU GPL v3 or later; see LICENSE.

This is a toolbox for planning and simulating in vivo focused-ultrasound (FUS)
neuromodulation in mice. It is built for researchers who are not
coders, so they should be able to lay out where to sonicate, then check what the
ultrasound field and heating look like through a mouse skull.

The repository contains two companion packages:


    - fus_planner decides where to sonicate. Given a mouse MRI/CT/atlas and a
    transducer's calibration, it computes the focal plane, a set of
    non-overlapping focal spots that cover the whole brain or a chosen Allen-CCF
    region, and a coordinate table referenced to anatomical landmarks.
    
    - fus_simulator shows what happens when you sonicate. It runs k-Wave
    acoustic simulations for a chosen transducer, calibrates the drive pressure to
    hit a target focal pressure, propagates the beam through a CT-derived skull,
    and estimates the resulting heating and thermal dose.


The two are designed to hand off to each other: the planner exports focal-spot
coordinates, and the simulator can import them to model the field at each spot.

We acknowledge the use of Claude (Anthropic) for the assistance in developing this software. 
Physical constants and methods are cited in REFERENCES.md.

For more details on each programme, read the README file in their respective folders.
