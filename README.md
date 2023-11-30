# Simulating the Universe using Diffusion

This repository contains my notebooks for my diffusion models for my master's project, see below...

## Project Summary

### Problem: The Universe is hard to simulate

To test the physics of different theories of the universe, we need simulations. To get simulations, we need supercomputers... Or do we? Generative AI is incredibly powerful and we're going to use diffusion to speed up the theoretical testing of physics! 

### Dataset: Open-source simulations

This project uses the CAMELs CMD dataset which includes over 150,000 2D simulations of universe. Each image is an universe which has different starting parameters which determine how the universe evolves. Determining how the universe evolves can be done in two ways:

- N-body
- Hydrodynamical ($M_{gas}$, $V_{gas}$, $T$...)

Hydrodynamical simulations are more physically realistic than N-body simulations but are much more computationally complex to simulate, typically requiring supercomputers.

<img src="Figures\Example fields.jpg" width="600"/>

Perhaps there's a way to convert the simple N-body simulations to realistic Hydrodynamical maps? 🤔 

## Model: Guassian Diffusion

A Diffusion model starts with a image of random noise and the model learns from the training dataset to slowly remove sections of the noise in the image until a clear denoised image is generated.

<img src="Figures\Novel_N_body_Diffusion_Gif.jpg" width="600"/>

Since the starting image is random, it acts as a seed

<img src="Figures\Conditon Diffusion Sample Fields.jpg" width="600"/>
